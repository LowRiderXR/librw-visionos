#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../rwbase.h"
#include "../rwerror.h"
#include "../rwplg.h"
#include "../rwrender.h"
#include "../rwengine.h"
#include "../rwpipeline.h"
#include "../rwobjects.h"
#ifdef RW_OPENGL
#include "rwgl3.h"
#include "rwgl3shader.h"

#include "rwgl3impl.h"

#ifdef LIBRW_VISIONOS
// Draw-call counter for the [vc-frame] probe. drawInst_simple is the ONE glDrawElements
// funnel for all atomic pipelines (default, matfx, skin -- they all go through drawInst),
// so counting here catches every world draw, including the PS2 alpha-test emulation which
// issues TWO draws per alpha-blended instance. Split by eye pass so the per-eye submission
// cost is visible. Storage and per-frame reset live in visionos.cpp.
extern "C" int vc_in_stereo_eye(void);   // 0 = mono/outside the loop, 1 = eye 0, 2 = eye 1
extern "C" unsigned g_vcDraws[3];
extern "C" unsigned g_vcTris[3];
// Per-draw overhead split (VC_DRAW_PROFILE=1): flushCache() -- GL state diffing plus the
// uniform registry walk -- against the bare glDrawElements. Two clock reads per draw, so
// it is gated; see visionos.cpp.
#include <mach/mach_time.h>
extern "C" int vc_draw_profile(void);
extern "C" uint64_t g_vcFlushTicks, g_vcDrawTicks;
// Finer split of the ~2.2 us per draw that sit OUTSIDE drawInst_simple (VAOs were
// measured and did NOT help, so the attribute setup is not it):
//   VTX = setupVertexInput + teardownVertexInput (per atomic)
//   PRE = per-atomic prologue: setWorldMatrix + lightingCB
//   MAT = per-mesh material/texture/shader selection before each draw
// eyes - (vtx+pre+mat+flush+gldraw) = what is left in reVC above librw.
extern "C" uint64_t g_vcVtxTicks, g_vcPreTicks, g_vcMatTicks;
#endif

namespace rw {
namespace gl3 {

#define MAX_LIGHTS

void
drawInst_simple(InstanceDataHeader *header, InstanceData *inst)
{
#ifdef LIBRW_VISIONOS
	{
		int et = vc_in_stereo_eye();
		if(et < 0 || et > 2) et = 0;
		g_vcDraws[et]++;
		g_vcTris[et] += (header->primType == GL_TRIANGLE_STRIP)
		              ? (inst->numIndex >= 2 ? inst->numIndex - 2 : 0)
		              : inst->numIndex / 3;
	}
	static int prof = -1;
	if(prof < 0) prof = vc_draw_profile();
	if(prof){
		uint64_t t0 = mach_absolute_time();
		flushCache();
		uint64_t t1 = mach_absolute_time();
		glDrawElements(header->primType, inst->numIndex,
		               GL_UNSIGNED_SHORT, (void*)(uintptr)inst->offset);
		uint64_t t2 = mach_absolute_time();
		g_vcFlushTicks += t1 - t0;
		g_vcDrawTicks  += t2 - t1;
		return;
	}
#endif
	flushCache();
	glDrawElements(header->primType, inst->numIndex,
	               GL_UNSIGNED_SHORT, (void*)(uintptr)inst->offset);
}

// Emulate PS2 GS alpha test FB_ONLY case: failed alpha writes to frame- but not to depth buffer
void
drawInst_GSemu(InstanceDataHeader *header, InstanceData *inst)
{
	uint32 hasAlpha;
	int alphafunc, alpharef, gsalpharef;
	int zwrite;
	hasAlpha = getAlphaBlend();
	if(hasAlpha){
		zwrite = rw::GetRenderState(rw::ZWRITEENABLE);
		alphafunc = rw::GetRenderState(rw::ALPHATESTFUNC);
		if(zwrite){
			alpharef = rw::GetRenderState(rw::ALPHATESTREF);
			gsalpharef = rw::GetRenderState(rw::GSALPHATESTREF);

			SetRenderState(rw::ALPHATESTFUNC, rw::ALPHAGREATEREQUAL);
			SetRenderState(rw::ALPHATESTREF, gsalpharef);
			drawInst_simple(header, inst);
			SetRenderState(rw::ALPHATESTFUNC, rw::ALPHALESS);
			SetRenderState(rw::ZWRITEENABLE, 0);
			drawInst_simple(header, inst);
			SetRenderState(rw::ZWRITEENABLE, 1);
			SetRenderState(rw::ALPHATESTFUNC, alphafunc);
			SetRenderState(rw::ALPHATESTREF, alpharef);
		}else{
			SetRenderState(rw::ALPHATESTFUNC, rw::ALPHAALWAYS);
			drawInst_simple(header, inst);
			SetRenderState(rw::ALPHATESTFUNC, alphafunc);
		}
	}else
		drawInst_simple(header, inst);
}

void
drawInst(InstanceDataHeader *header, InstanceData *inst)
{
	if(rw::GetRenderState(rw::GSALPHATEST))
		drawInst_GSemu(header, inst);
	else
		drawInst_simple(header, inst);
}


void
setAttribPointers(AttribDesc *attribDescs, int32 numAttribs)
{
	AttribDesc *a;
	for(a = attribDescs; a != &attribDescs[numAttribs]; a++){
		glEnableVertexAttribArray(a->index);
		glVertexAttribPointer(a->index, a->size, a->type, a->normalized,
		                      a->stride, (void*)(uint64)a->offset);
	}
}

void
disableAttribPointers(AttribDesc *attribDescs, int32 numAttribs)
{
	AttribDesc *a;
	for(a = attribDescs; a != &attribDescs[numAttribs]; a++)
		glDisableVertexAttribArray(a->index);
}

void
setupVertexInput(InstanceDataHeader *header)
{
#ifdef RW_GL_USE_VAOS
	glBindVertexArray(header->vao);
#else
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, header->ibo);
	glBindBuffer(GL_ARRAY_BUFFER, header->vbo);
	setAttribPointers(header->attribDesc, header->numAttribs);
#endif
}

void
teardownVertexInput(InstanceDataHeader *header)
{
#ifndef RW_GL_USE_VAOS
	disableAttribPointers(header->attribDesc, header->numAttribs);
#endif
}

int32
lightingCB(Atomic *atomic)
{
	WorldLights lightData;
	Light *directionals[8];
	Light *locals[8];
	lightData.directionals = directionals;
	lightData.numDirectionals = 8;
	lightData.locals = locals;
	lightData.numLocals = 8;

	if(atomic->geometry->flags & rw::Geometry::LIGHT){
		((World*)engine->currentWorld)->enumerateLights(atomic, &lightData);
		if((atomic->geometry->flags & rw::Geometry::NORMALS) == 0){
			// Get rid of lights that need normals when we don't have any
			lightData.numDirectionals = 0;
			lightData.numLocals = 0;
		}
		return setLights(&lightData);
	}else{
		memset(&lightData, 0, sizeof(lightData));
		return setLights(&lightData);
	}
}

void
defaultRenderCB(Atomic *atomic, InstanceDataHeader *header)
{
	Material *m;

	uint32 flags = atomic->geometry->flags;
#ifdef LIBRW_VISIONOS
	static int prof = -1;
	if(prof < 0) prof = vc_draw_profile();
	uint64_t tA = prof ? mach_absolute_time() : 0;
#endif
	setWorldMatrix(atomic->getFrame()->getLTM());
	int32 vsBits = lightingCB(atomic);
#ifdef LIBRW_VISIONOS
	uint64_t tB = prof ? mach_absolute_time() : 0;
#endif

	setupVertexInput(header);
#ifdef LIBRW_VISIONOS
	uint64_t tC = prof ? mach_absolute_time() : 0;
	if(prof){ g_vcPreTicks += tB - tA; g_vcVtxTicks += tC - tB; }
#endif

	InstanceData *inst = header->inst;
	int32 n = header->numMeshes;

	while(n--){
		m = inst->material;
#ifdef LIBRW_VISIONOS
		uint64_t tM = prof ? mach_absolute_time() : 0;
#endif

		setMaterial(flags, m->color, m->surfaceProps);

		setTexture(0, m->texture);

		rw::SetRenderState(VERTEXALPHA, inst->vertexAlpha || m->color.alpha != 0xFF);

		if((vsBits & VSLIGHT_MASK) == 0){
			if(getAlphaTest())
				defaultShader->use();
			else
				defaultShader_noAT->use();
		}else{
			if(getAlphaTest())
				defaultShader_fullLight->use();
			else
				defaultShader_fullLight_noAT->use();
		}
#ifdef LIBRW_VISIONOS
		if(prof) g_vcMatTicks += mach_absolute_time() - tM;
#endif

		drawInst(header, inst);
		inst++;
	}
#ifdef LIBRW_VISIONOS
	uint64_t tT = prof ? mach_absolute_time() : 0;
#endif
	teardownVertexInput(header);
#ifdef LIBRW_VISIONOS
	if(prof) g_vcVtxTicks += mach_absolute_time() - tT;
#endif
}


}
}

#endif

