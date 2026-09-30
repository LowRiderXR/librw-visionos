#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../rwbase.h"
#include "../rwerror.h"
#include "../rwplg.h"
#include "../rwrender.h"
#include "../rwpipeline.h"
#include "../rwobjects.h"
#include "../rwengine.h"
#ifdef RW_OPENGL
#include "rwgl3.h"
#include "rwgl3impl.h"
#include "rwgl3shader.h"

#ifdef LIBRW_VISIONOS
// Draw counter for the IMMEDIATE paths (im2d/im3d), split by eye pass like g_vcDraws.
// g_vcDraws counts only drawInst_simple (the atomic pipelines), so sprites, particles,
// coronas, clouds, rain and HUD were invisible to the profiler. This share decides how
// much of the frame multiview can touch at all: im2d geometry is CPU-projected per eye
// (CalcScreenCoors bakes the eye into the VERTICES, not into a matrix), so one
// amplified pass cannot render it correctly without further work. Indices:
// [0] outside the eye loop, [1] eye 0, [2] eye 1. Storage/reset in visionos.cpp.
extern "C" int vc_in_stereo_eye(void);
extern "C" unsigned g_vcImmDraws[3];
static inline void vcCountImmDraw(void)
{
	int et = vc_in_stereo_eye();
	if(et < 0 || et > 2) et = 0;
	g_vcImmDraws[et]++;
}
#else
static inline void vcCountImmDraw(void) {}
#endif

namespace rw {
namespace gl3 {

uint32 im2DVbo, im2DIbo;
#ifdef RW_GL_USE_VAOS
uint32 im2DVao;
#endif

Shader *im2dOverrideShader;

static int32 u_xform;

#define STARTINDICES 10000
#define STARTVERTICES 10000

static Shader *im2dShader;
static AttribDesc im2dattribDesc[3] = {
	{ ATTRIB_POS,        GL_FLOAT,         GL_FALSE, 4,
		sizeof(Im2DVertex), 0 },
	{ ATTRIB_COLOR,      GL_UNSIGNED_BYTE, GL_TRUE,  4,
		sizeof(Im2DVertex), offsetof(Im2DVertex, r) },
	{ ATTRIB_TEXCOORDS0, GL_FLOAT,         GL_FALSE, 2,
		sizeof(Im2DVertex), offsetof(Im2DVertex, u) },
};

static int primTypeMap[] = {
	GL_POINTS,	// invalid
	GL_LINES,
	GL_LINE_STRIP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN,
	GL_POINTS
};

void
openIm2D(void)
{
	// must already be registered by device. we just need the value
	u_xform = registerUniform("u_xform", UNIFORM_VEC4);

#include "shaders/im2d_gl.inc"
#include "shaders/simple_fs_gl.inc"
	const char *vs[] = { shaderDecl, header_vert_src, im2d_vert_src, nil };
	const char *fs[] = { shaderDecl, header_frag_src, simple_frag_src, nil };
	im2dShader = Shader::create(vs, fs, "im2d");
	assert(im2dShader);

	glGenBuffers(1, &im2DIbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im2DIbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, STARTINDICES*2, nil, GL_STREAM_DRAW);

	glGenBuffers(1, &im2DVbo);
	glBindBuffer(GL_ARRAY_BUFFER, im2DVbo);
	glBufferData(GL_ARRAY_BUFFER, STARTVERTICES*sizeof(Im2DVertex), nil, GL_STREAM_DRAW);

#ifdef RW_GL_USE_VAOS
	glGenVertexArrays(1, &im2DVao);
	glBindVertexArray(im2DVao);
	setAttribPointers(im2dattribDesc, 3);
#endif
}

void
closeIm2D(void)
{
	glDeleteBuffers(1, &im2DIbo);
	glDeleteBuffers(1, &im2DVbo);
#ifdef RW_GL_USE_VAOS
	glDeleteVertexArrays(1, &im2DVao);
#endif
	im2dShader->destroy();
	im2dShader = nil;
}

static Im2DVertex tmpprimbuf[3];

void
im2DRenderLine(void *vertices, int32 numVertices, int32 vert1, int32 vert2)
{
	Im2DVertex *verts = (Im2DVertex*)vertices;
	tmpprimbuf[0] = verts[vert1];
	tmpprimbuf[1] = verts[vert2];
	im2DRenderPrimitive(PRIMTYPELINELIST, tmpprimbuf, 2);
}

void
im2DRenderTriangle(void *vertices, int32 numVertices, int32 vert1, int32 vert2, int32 vert3)
{
	Im2DVertex *verts = (Im2DVertex*)vertices;
	tmpprimbuf[0] = verts[vert1];
	tmpprimbuf[1] = verts[vert2];
	tmpprimbuf[2] = verts[vert3];
	im2DRenderPrimitive(PRIMTYPETRILIST, tmpprimbuf, 3);
}

void
im2DSetXform(void)
{
	GLfloat xform[4];
	Camera *cam;
	cam = (Camera*)engine->currentCamera;
	xform[0] = 2.0f/cam->frameBuffer->width;
	xform[1] = -2.0f/cam->frameBuffer->height;
	xform[2] = -1.0f;
	xform[3] = 1.0f;
	setUniform(u_xform, xform);
//	glUniform4fv(currentShader->uniformLocations[u_xform], 1, xform);
}

void
im2DRenderPrimitive(PrimitiveType primType, void *vertices, int32 numVertices)
{
#ifdef RW_GL_USE_VAOS
	glBindVertexArray(im2DVao);
#endif

	glBindBuffer(GL_ARRAY_BUFFER, im2DVbo);
	glBufferData(GL_ARRAY_BUFFER, STARTVERTICES*sizeof(Im2DVertex), nil, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numVertices*sizeof(Im2DVertex), vertices);

	if(im2dOverrideShader)
		im2dOverrideShader->use();
	else
		im2dShader->use();
#ifndef RW_GL_USE_VAOS
	setAttribPointers(im2dattribDesc, 3);
#endif

	im2DSetXform();

	flushCache();
	vcCountImmDraw();
	glDrawArrays(primTypeMap[primType], 0, numVertices);
#ifndef RW_GL_USE_VAOS
	disableAttribPointers(im2dattribDesc, 3);
#endif
}

void
im2DRenderIndexedPrimitive(PrimitiveType primType,
	void *vertices, int32 numVertices,
	void *indices, int32 numIndices)
{
#ifdef RW_GL_USE_VAOS
	glBindVertexArray(im2DVao);
#endif

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im2DIbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, STARTINDICES*2, nil, GL_STREAM_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, numIndices*2, indices);

	glBindBuffer(GL_ARRAY_BUFFER, im2DVbo);
	glBufferData(GL_ARRAY_BUFFER, STARTVERTICES*sizeof(Im2DVertex), nil, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numVertices*sizeof(Im2DVertex), vertices);

	if(im2dOverrideShader)
		im2dOverrideShader->use();
	else
		im2dShader->use();
#ifndef RW_GL_USE_VAOS
	setAttribPointers(im2dattribDesc, 3);
#endif

	im2DSetXform();

	flushCache();
	vcCountImmDraw();
	glDrawElements(primTypeMap[primType], numIndices,
	               GL_UNSIGNED_SHORT, nil);
#ifndef RW_GL_USE_VAOS
	disableAttribPointers(im2dattribDesc, 3);
#endif
}


// Im3D


static Shader *im3dShader;
static AttribDesc im3dattribDesc[3] = {
	{ ATTRIB_POS,        GL_FLOAT,         GL_FALSE, 3,
		sizeof(Im3DVertex), 0 },
	{ ATTRIB_COLOR,      GL_UNSIGNED_BYTE, GL_TRUE,  4,
		sizeof(Im3DVertex), offsetof(Im3DVertex, r) },
	{ ATTRIB_TEXCOORDS0, GL_FLOAT,         GL_FALSE, 2,
		sizeof(Im3DVertex), offsetof(Im3DVertex, u) },
};
static uint32 im3DVbo, im3DIbo;
#ifdef RW_GL_USE_VAOS
static uint32 im3DVao;
#endif
static int32 num3DVertices;	// not actually needed here

// u_im3dPull: view-space depth pull for world-space sprites (see im3d.vert). Uploaded on
// every im3DTransform; 0 unless the visionOS corona path sets it around its quads.
static int32 u_im3dPull;
static float g_im3dPull[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#ifdef LIBRW_VISIONOS
extern "C" void vc_im3d_pull(float delta) { g_im3dPull[0] = delta; }

// Device probe (VC_CORONA_DIAG=1): after the uniforms of a pulled im3d draw were flushed,
// read the value back from the program that is actually bound. Tells whether the pull
// reaches the shader at all (location, program, value) -- the CPU side looks right.
static void
vcIm3dPullProbe(void)
{
	static int enabled = -1;
	if(enabled < 0){
		const char *e = getenv("VC_CORONA_DIAG");
		enabled = (e && e[0] == '1') ? 1 : 0;
	}
	if(!enabled || g_im3dPull[0] == 0.0f) return;
	static int counter = 0;
	if(counter++ % 300 != 0) return;
	GLint loc = currentShader->uniformLocations[u_im3dPull];
	float v[4] = { -99.0f, -99.0f, -99.0f, -99.0f };
	if(loc >= 0) glGetUniformfv(currentShader->program, loc, v);
	GLint bound = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &bound);
	const char *which = currentShader == im3dShader ? "mono" :
		(im3dShader->mv && currentShader == im3dShader->mv) ? "twin" : "OTHER";
	printf("[vc-im3d] pull probe: shader=%s program=%u bound=%d loc=%d want=%.3f inProgram=%.3f serialSynced=%d glErr=0x%x\n",
	       which, currentShader->program, bound, loc, g_im3dPull[0], v[0],
	       currentShader->serialNums[u_im3dPull] == uniformRegistry.uniforms[u_im3dPull].serialNum,
	       glGetError());
}
#endif

void
openIm3D(void)
{
	// registered by the device at init; we just need the id
	u_im3dPull = registerUniform("u_im3dPull", UNIFORM_VEC4);
#include "shaders/im3d_gl.inc"
#include "shaders/simple_fs_gl.inc"
	const char *vs[] = { shaderDecl, header_vert_src, im3d_vert_src, nil };
	const char *fs[] = { shaderDecl, header_frag_src, simple_frag_src, nil };
	im3dShader = Shader::create(vs, fs, "im3d");
	assert(im3dShader);
#ifdef LIBRW_VISIONOS
	printf("[vc-im3d] u_im3dPull id=%d location mono=%d twin=%d\n", u_im3dPull,
	       im3dShader->uniformLocations[u_im3dPull],
	       im3dShader->mv ? im3dShader->mv->uniformLocations[u_im3dPull] : -2);
#endif

	glGenBuffers(1, &im3DIbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im3DIbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, STARTINDICES*2, nil, GL_STREAM_DRAW);

	glGenBuffers(1, &im3DVbo);
	glBindBuffer(GL_ARRAY_BUFFER, im3DVbo);
	glBufferData(GL_ARRAY_BUFFER, STARTVERTICES*sizeof(Im3DVertex), nil, GL_STREAM_DRAW);

#ifdef RW_GL_USE_VAOS
	glGenVertexArrays(1, &im3DVao);
	glBindVertexArray(im3DVao);
	setAttribPointers(im3dattribDesc, 3);
#endif
}

void
closeIm3D(void)
{
	glDeleteBuffers(1, &im3DIbo);
	glDeleteBuffers(1, &im3DVbo);
#ifdef RW_GL_USE_VAOS
	glDeleteVertexArrays(1, &im3DVao);
#endif
	im3dShader->destroy();
	im3dShader = nil;
}

void
im3DTransform(void *vertices, int32 numVertices, Matrix *world, uint32 flags)
{
	if(world == nil){
		static Matrix ident;
		ident.setIdentity();
		world = &ident;
	}
	setWorldMatrix(world);
	im3dShader->use();
	setUniform(u_im3dPull, g_im3dPull);

	if((flags & im3d::VERTEXUV) == 0)
		SetRenderStatePtr(TEXTURERASTER, nil);

#ifdef RW_GL_USE_VAOS
	glBindVertexArray(im2DVao);
#endif

	glBindBuffer(GL_ARRAY_BUFFER, im3DVbo);
	glBufferData(GL_ARRAY_BUFFER, STARTVERTICES*sizeof(Im3DVertex), nil, GL_STREAM_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numVertices*sizeof(Im3DVertex), vertices);
#ifndef RW_GL_USE_VAOS
	setAttribPointers(im3dattribDesc, 3);
#endif
	num3DVertices = numVertices;
}

void
im3DRenderPrimitive(PrimitiveType primType)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im3DIbo);

	flushCache();
	vcCountImmDraw();
	glDrawArrays(primTypeMap[primType], 0, num3DVertices);
}

void
im3DRenderIndexedPrimitive(PrimitiveType primType, void *indices, int32 numIndices)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im3DIbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, STARTINDICES*2, nil, GL_STREAM_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, numIndices*2, indices);

	flushCache();
#ifdef LIBRW_VISIONOS
	vcIm3dPullProbe();
#endif
	vcCountImmDraw();
	glDrawElements(primTypeMap[primType], numIndices,
	               GL_UNSIGNED_SHORT, nil);
}

void
im3DEnd(void)
{
#ifndef RW_GL_USE_VAOS
	disableAttribPointers(im3dattribDesc, 3);
#endif
}

}
}

#endif
