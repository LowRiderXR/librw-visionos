#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../rwbase.h"
#include "../rwerror.h"
#include "../rwplg.h"
#include "../rwrender.h"
#include "../rwengine.h"
#include "../rwpipeline.h"
#include "../rwobjects.h"
#include "../rwanim.h"
#include "../rwplugins.h"

#include "rwgl3.h"
#include "rwgl3shader.h"
#include "rwgl3plg.h"

#include "rwgl3impl.h"

namespace rw {
namespace gl3 {

#ifdef RW_OPENGL

static Shader *skinShader, *skinShader_noAT;
static Shader *skinShader_fullLight, *skinShader_fullLight_noAT;
static int32 u_boneMatrices;

void
skinInstanceCB(Geometry *geo, InstanceDataHeader *header, bool32 reinstance)
{
	AttribDesc *attribs, *a;

	bool isPrelit = !!(geo->flags & Geometry::PRELIT);
	bool hasNormals = !!(geo->flags & Geometry::NORMALS);

	if(!reinstance){
		AttribDesc tmpAttribs[14];
		uint32 stride;

		//
		// Create attribute descriptions
		//
		a = tmpAttribs;
		stride = 0;

		// Positions
		a->index = ATTRIB_POS;
		a->size = 3;
		a->type = GL_FLOAT;
		a->normalized = GL_FALSE;
		a->offset = stride;
		stride += 12;
		a++;

		// Normals
		// TODO: compress
		if(hasNormals){
			a->index = ATTRIB_NORMAL;
			a->size = 3;
			a->type = GL_FLOAT;
			a->normalized = GL_FALSE;
			a->offset = stride;
			stride += 12;
			a++;
		}

		// Prelighting
		if(isPrelit){
			a->index = ATTRIB_COLOR;
			a->size = 4;
			a->type = GL_UNSIGNED_BYTE;
			a->normalized = GL_TRUE;
			a->offset = stride;
			stride += 4;
			a++;
		}

		// Texture coordinates
		for(int32 n = 0; n < geo->numTexCoordSets; n++){
			a->index = ATTRIB_TEXCOORDS0+n;
			a->size = 2;
			a->type = GL_FLOAT;
			a->normalized = GL_FALSE;
			a->offset = stride;
			stride += 8;
			a++;
		}

		// Weights
		a->index = ATTRIB_WEIGHTS;
		a->size = 4;
		a->type = GL_FLOAT;
		a->normalized = GL_FALSE;
		a->offset = stride;
		stride += 16;
		a++;

		// Indices
		a->index = ATTRIB_INDICES;
		a->size = 4;
		a->type = GL_UNSIGNED_BYTE;
		a->normalized = GL_FALSE;
		a->offset = stride;
		stride += 4;
		a++;

		header->numAttribs = a - tmpAttribs;
		for(a = tmpAttribs; a != &tmpAttribs[header->numAttribs]; a++)
			a->stride = stride;
		header->attribDesc = rwNewT(AttribDesc, header->numAttribs, MEMDUR_EVENT | ID_GEOMETRY);
		memcpy(header->attribDesc, tmpAttribs,
		       header->numAttribs*sizeof(AttribDesc));

		//
		// Allocate vertex buffer
		//
		header->vertexBuffer = rwNewT(uint8, header->totalNumVertex*stride, MEMDUR_EVENT | ID_GEOMETRY);
		assert(header->vbo == 0);
		glGenBuffers(1, &header->vbo);
	}

	Skin *skin = Skin::get(geo);
	attribs = header->attribDesc;

	//
	// Fill vertex buffer
	//

	uint8 *verts = header->vertexBuffer;

	// Positions
	if(!reinstance || geo->lockedSinceInst&Geometry::LOCKVERTICES){
		for(a = attribs; a->index != ATTRIB_POS; a++)
			;
		instV3d(VERT_FLOAT3, verts + a->offset,
			geo->morphTargets[0].vertices,
			header->totalNumVertex, a->stride);
	}

	// Normals
	if(hasNormals && (!reinstance || geo->lockedSinceInst&Geometry::LOCKNORMALS)){
		for(a = attribs; a->index != ATTRIB_NORMAL; a++)
			;
		instV3d(VERT_FLOAT3, verts + a->offset,
			geo->morphTargets[0].normals,
			header->totalNumVertex, a->stride);
	}

	// Prelighting
	if(isPrelit && (!reinstance || geo->lockedSinceInst&Geometry::LOCKPRELIGHT)){
		for(a = attribs; a->index != ATTRIB_COLOR; a++)
			;
		instColor(VERT_RGBA, verts + a->offset,
			  geo->colors,
			  header->totalNumVertex, a->stride);
	}

	// Texture coordinates
	for(int32 n = 0; n < geo->numTexCoordSets; n++){
		if(!reinstance || geo->lockedSinceInst&(Geometry::LOCKTEXCOORDS<<n)){
			for(a = attribs; a->index != ATTRIB_TEXCOORDS0+n; a++)
				;
			instTexCoords(VERT_FLOAT2, verts + a->offset,
				geo->texCoords[n],
				header->totalNumVertex, a->stride);
		}
	}

	// Weights
	if(!reinstance){
		for(a = attribs; a->index != ATTRIB_WEIGHTS; a++)
			;
		float *w = skin->weights;
		instV4d(VERT_FLOAT4, verts + a->offset,
			(V4d*)w,
			header->totalNumVertex, a->stride);
	}

	// Indices
	if(!reinstance){
		for(a = attribs; a->index != ATTRIB_INDICES; a++)
			;
		// not really colors of course but what the heck
		instColor(VERT_RGBA, verts + a->offset,
			  (RGBA*)skin->indices,
			  header->totalNumVertex, a->stride);
	}

#ifdef RW_GL_USE_VAOS
	glBindVertexArray(header->vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, header->ibo);
#endif
	glBindBuffer(GL_ARRAY_BUFFER, header->vbo);
	glBufferData(GL_ARRAY_BUFFER, header->totalNumVertex*attribs[0].stride,
	             header->vertexBuffer, GL_STATIC_DRAW);
#ifdef RW_GL_USE_VAOS
	setAttribPointers(header->attribDesc, header->numAttribs);
	glBindVertexArray(0);
#endif
}

void
skinUninstanceCB(Geometry *geo, InstanceDataHeader *header)
{
	assert(0 && "can't uninstance");
}

static float skinMatrices[64*16];

void
uploadSkinMatrices(Atomic *a)
{
	int i;
	Skin *skin = Skin::get(a->geometry);
	Matrix *m = (Matrix*)skinMatrices;
	HAnimHierarchy *hier = Skin::getHierarchy(a);

	if(hier){
		Matrix *invMats = (Matrix*)skin->inverseMatrices;
		Matrix tmp;

		assert(skin->numBones == hier->numNodes);
		if(hier->flags & HAnimHierarchy::LOCALSPACEMATRICES){
			for(i = 0; i < hier->numNodes; i++){
				invMats[i].flags = 0;
				Matrix::mult(m, &invMats[i], &hier->matrices[i]);
				m++;
			}
		}else{
			Matrix invAtmMat;
			Matrix::invert(&invAtmMat, a->getFrame()->getLTM());
			for(i = 0; i < hier->numNodes; i++){
				invMats[i].flags = 0;
				Matrix::mult(&tmp, &hier->matrices[i], &invAtmMat);
				Matrix::mult(m, &invMats[i], &tmp);
				m++;
			}
		}
	}else{
		for(i = 0; i < skin->numBones; i++){
			m->setIdentity();
			m++;
		}
	}
	setUniform(u_boneMatrices, skinMatrices);
}

void
skinRenderCB(Atomic *atomic, InstanceDataHeader *header)
{
	Material *m;

	uint32 flags = atomic->geometry->flags;
	setWorldMatrix(atomic->getFrame()->getLTM());
	int32 vsBits = lightingCB(atomic);

	setupVertexInput(header);

	InstanceData *inst = header->inst;
	int32 n = header->numMeshes;

	uploadSkinMatrices(atomic);

	while(n--){
		m = inst->material;

		setMaterial(flags, m->color, m->surfaceProps);

		setTexture(0, m->texture);

		rw::SetRenderState(VERTEXALPHA, inst->vertexAlpha || m->color.alpha != 0xFF);

		if((vsBits & VSLIGHT_MASK) == 0){
			if(getAlphaTest())
				skinShader->use();
			else
				skinShader_noAT->use();
		}else{
			if(getAlphaTest())
				skinShader_fullLight->use();
			else
				skinShader_fullLight_noAT->use();
		}

		drawInst(header, inst);
		inst++;
	}
	teardownVertexInput(header);
}

#ifdef LIBRW_VISIONOS
// The two ped shader fixes below are needed by a SECOND pipeline as well: reVC's neo rim
// pipeline (src/extras/custompipes_gl.cpp) compiles its own copy of simple.frag plus its own
// skin vertex shader, and peds get that pipeline UNCONDITIONALLY
// (CPedModelInfo::SetClump -> CustomPipes::AttachRimPipe). Without sharing the defines, both
// fixes silently stopped applying whenever "CHARAKTER KANTEN LICHT" (NeoRimLight) was on --
// device-confirmed: the NPC discoloration came back with the effect enabled. Exported here so
// value and default live in ONE place.
//
// VC_SKIN_FOG: skinned models fed clip-Z to DoFog (skin.vert / neoRimSkin.vert) while
// buildings (default.vert) and neoRim.vert feed clip-W. Our per-eye rebuilt projection makes
// clip-Z the wrong fog input. =z restores the original for A/B, =o isolates fog entirely.
extern "C" const char *vc_skin_fog_define(void)
{
	static const char *def = NULL;
	if(!def){
		const char *e = getenv("VC_SKIN_FOG");
		if(e && e[0] == 'z')       def = "#define VC_SKINFOG z\n";      // original (desktop)
		else if(e && e[0] == 'o')  def = "#define VC_SKIN_NOFOG 1\n";   // fog isolation test
		else                       def = "#define VC_SKINFOG w\n";      // default fix (view depth)
	}
	return def;
}
// VC_SKIN_LODBIAS: peds pack face+body+clothes in one atlas; at low pixel coverage (distance
// / VR angular resolution) the trilinear mip averages adjacent UV regions -> discoloration.
// Negative bias = sharper mip, trades that bleed for a little more shimmer. =0 disables.
extern "C" const char *vc_skin_lodbias_define(void)
{
	static const char *def = NULL;
	if(!def){
		const char *e = getenv("VC_SKIN_LODBIAS");
		float bias = (e && e[0]) ? (float)atof(e) : -2.5f;
		if(bias != 0.0f){
			static char buf[64];
			snprintf(buf, sizeof(buf), "#define VC_SKIN_LODBIAS (%f)\n", bias);
			def = buf;
		}else
			def = "";
	}
	return def;
}
#endif

static void*
skinOpen(void *o, int32, int32)
{
	skinGlobals.pipelines[PLATFORM_GL3] = makeSkinPipeline();

#include "shaders/simple_fs_gl.inc"
#include "shaders/skin_gl.inc"
#ifdef LIBRW_VISIONOS
	// Fog + mip-bias defines now come from the shared getters above (same value for this
	// pipeline and reVC's neo rim pipeline).
	const char *vcSkinFogDef = vc_skin_fog_define();
	const char *vs[] = { shaderDecl, vcSkinFogDef, header_vert_src, skin_vert_src, nil };
	const char *vs_fullLight[] = { shaderDecl, vcSkinFogDef, "#define DIRECTIONALS\n#define POINTLIGHTS\n#define SPOTLIGHTS\n", header_vert_src, skin_vert_src, nil };
#else
	const char *vs[] = { shaderDecl, header_vert_src, skin_vert_src, nil };
	const char *vs_fullLight[] = { shaderDecl, "#define DIRECTIONALS\n#define POINTLIGHTS\n#define SPOTLIGHTS\n", header_vert_src, skin_vert_src, nil };
#endif
#ifdef LIBRW_VISIONOS
	const char *vcSkinLodDef = vc_skin_lodbias_define();
	const char *fs[]     = { shaderDecl, vcSkinLodDef, header_frag_src, simple_frag_src, nil };
	const char *fs_noAT[] = { shaderDecl, vcSkinLodDef, "#define NO_ALPHATEST\n", header_frag_src, simple_frag_src, nil };
#else
	const char *fs[] = { shaderDecl, header_frag_src, simple_frag_src, nil };
	const char *fs_noAT[] = { shaderDecl, "#define NO_ALPHATEST\n", header_frag_src, simple_frag_src, nil };
#endif

	skinShader = Shader::create(vs, fs);
	assert(skinShader);
	skinShader_noAT = Shader::create(vs, fs_noAT);
	assert(skinShader_noAT);

	skinShader_fullLight = Shader::create(vs_fullLight, fs);
	assert(skinShader_fullLight);
	skinShader_fullLight_noAT = Shader::create(vs_fullLight, fs_noAT);
	assert(skinShader_fullLight_noAT);

	return o;
}

static void*
skinClose(void *o, int32, int32)
{
	((ObjPipeline*)skinGlobals.pipelines[PLATFORM_GL3])->destroy();
	skinGlobals.pipelines[PLATFORM_GL3] = nil;

	skinShader->destroy();
	skinShader = nil;
	skinShader_noAT->destroy();
	skinShader_noAT = nil;
	skinShader_fullLight->destroy();
	skinShader_fullLight = nil;
	skinShader_fullLight_noAT->destroy();
	skinShader_fullLight_noAT = nil;

	return o;
}

void
initSkin(void)
{
	u_boneMatrices = registerUniform("u_boneMatrices", UNIFORM_MAT4, 64);

	Driver::registerPlugin(PLATFORM_GL3, 0, ID_SKIN,
	                       skinOpen, skinClose);
}

ObjPipeline*
makeSkinPipeline(void)
{
	ObjPipeline *pipe = ObjPipeline::create();
	pipe->instanceCB = skinInstanceCB;
	pipe->uninstanceCB = skinUninstanceCB;
	pipe->renderCB = skinRenderCB;
	pipe->pluginID = ID_SKIN;
	pipe->pluginData = 1;
	return pipe;
}

#else

void initSkin(void) { }

#endif

}
}

