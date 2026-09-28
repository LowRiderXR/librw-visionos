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
#ifdef RW_OPENGL

#include "rwgl3.h"
#include "rwgl3shader.h"
#include "rwgl3impl.h"

#define PLUGIN_ID 0

#ifdef LIBRW_VISIONOS
// Provided by the visionOS platform layer (src/skel/visionos). setFrameBuffer()
// consults these to redirect the surfaceless default framebuffer (fbo 0) to the
// external EGLImage-backed FBO. Named explicitly so the redirect is intentional.
extern "C" bool vc_use_external_framebuffer(void);
extern "C" unsigned int vc_external_framebuffer(void);
// Attach librw's shared depth renderbuffer to BOTH back-buffer FBOs at once.
// Needed because the redirect swaps FBOs per frame while librw tracks a single
// fboMate on the shared CAMERA raster: its "all good" fast path would otherwise
// leave the other back buffer without a depth attachment.
extern "C" void vc_attach_depth_renderbuffer(unsigned int rbo);
// Single source for the render-target size (from VISIONOS_SCREEN_* in
// visionos.cpp). Used for the one hardcoded video mode below.
extern "C" void vc_screen_size(int *w, int *h);
// Camera matrix override seam (stereo injection point) + VC_MATRIX_TEST driver.
// beginUpdate consumes these after computing its own view/proj. Defined in
// src/skel/visionos/visionos.cpp.
extern "C" int  vc_matrix_test_mode(void);
extern "C" void vc_set_view_matrix(const float m[16]);
extern "C" void vc_set_projection_matrix(const float m[16]);
extern "C" void vc_set_matrix_override(int active);
extern "C" int  vc_matrix_override_active(void);
extern "C" void vc_get_view_matrix(float m[16]);
extern "C" void vc_get_projection_matrix(float m[16]);
extern "C" int  vc_view_compose_active(void);   // 1 = supplied view is a head offset -> V_final = offset * V_game
extern "C" int  vc_render_mode(void);           // 1 = VC_MODE_STEREO (sprite head-view fix scope)
// Phase 5.5 stereo target (visionos_angle.mm): lazily create the 2D-array render
// target and get a slice's GL FBO. The two eye passes below redirect to these.
extern "C" bool         vcrt_stereo_ensure(void);
extern "C" unsigned int vc_stereo_eye_fbo(int eye);
extern "C" void         vcrt_mv_pairs_report(void);   // 5.0b/1: one-shot multiview program-pair summary
extern "C" unsigned int vc_multiview_fbo(void);        // 5.0b/2: the one FBO whose binding selects the twins (0 = none yet)
extern "C" void         vc_multiview_fbo_set(unsigned int fbo);
extern "C" int          vc_mv_pairs_mode(void);
extern "C" int          vc_multiview_active(void);    // 5.0b/5: one-pass render active (getter semantics)
extern "C" void         vc_stereo_msaa_resolve_pending(void);   // VC_MSAA: resolve last eye's MSAA into its slice
// Phase 5.6: real per-eye compositor matrices, already in librw convention (LH /
// +Z / clip depth -1..1), translations in METRES. Mirror of vc_stereo_eye_matrices_t
// in VCPlatform.h -- the struct layout MUST match. Implemented in VCRendererStub.mm,
// which is linked only into the visionOS app, so this lives entirely under
// LIBRW_VISIONOS; the macOS reference build never references it.
#include <simd/simd.h>
typedef struct vc_stereo_eye_matrices_t {
	simd_float4x4 view[2];        // world->eye, librw convention, metres
	simd_float4x4 projection[2];  // librw clip (REVERSE-Z: near=+1, far=-1) -- see eye pass
	uint32_t      valid;          // 0 = not populated (cinema / not ready yet)
} vc_stereo_eye_matrices_t;
extern "C" bool vc_get_stereo_eye_matrices(vc_stereo_eye_matrices_t *out);
#endif

#ifdef LIBRW_VISIONOS
extern "C" unsigned g_vcTexBinds;   // see visionos.cpp (per-draw overhead split)
// Splitting the eye-pass boundary. MEASURED on device: with VC_MSAA=2 the wait sits in
// eyPre (the resolve blit), with VC_MSAA=0 it moves to eyClr (the full-screen clear) --
// SAME total either way. So the boundary is not an expensive operation, it is a
// SYNCHRONISATION POINT: whichever call first touches the render target absorbs the wait.
// (eyBind is always ~0: ANGLE binds lazily and only creates the Metal encoder on the first
// real operation.) Slot ids match the VC_SC_* enum in visionos.cpp.
extern "C" void vc_scene_begin(int id);
extern "C" void vc_scene_end(int id);
extern "C" void vc_scene_set_eye(int eye);
#define VC_SC_EYBIND 13
#define VC_SC_EYCLR  14
#define VC_SC_EYMTX  15
#define VC_SC_EYPRE  16   // prologue: vcrt_stereo_ensure + vc_stereo_eye_fbo (MSAA resolve!)

#endif

namespace rw {
namespace gl3 {

GlGlobals glGlobals;

Gl3Caps gl3Caps;
// terrible hack for GLES
bool32 needToReadBackTextures;

int32   alphaFunc;
float32 alphaRef;

struct UniformState
{
	float32 alphaRefLow;
	float32 alphaRefHigh;
	int32   pad[2];

	float32 fogStart;
	float32 fogEnd;
	float32 fogRange;
	float32 fogDisable;

	RGBAf   fogColor;
};

struct UniformScene
{
	float32 proj[16];
	float32 view[16];
};

#define MAX_LIGHTS 8

struct UniformObject
{
	RawMatrix    world;
	RGBAf        ambLight;
	struct {
		float type;
		float radius;
		float minusCosAngle;
		float hardSpot;
	} lightParams[MAX_LIGHTS];
	V4d lightPosition[MAX_LIGHTS];
	V4d lightDirection[MAX_LIGHTS];
	RGBAf lightColor[MAX_LIGHTS];
};

const char *shaderDecl120 =
"#version 120\n"
"#define GL2\n"
"#define texture texture2D\n"
"#define VSIN(index) attribute\n"
"#define VSOUT varying\n"
"#define FSIN varying\n"
"#define FRAGCOLOR(c) (gl_FragColor = c)\n";
const char *shaderDecl330 =
"#version 330\n"
"#define VSIN(index) layout(location = index) in\n"
"#define VSOUT out\n"
"#define FSIN in\n"
"#define FRAGCOLOR(c) (fragColor = c)\n";
const char *shaderDecl100es =
"#version 100\n"
"#define GL2\n"
"#define texture texture2D\n"
"#define VSIN(index) attribute\n"
"#define VSOUT varying\n"
"#define FSIN varying\n"
"#define FRAGCOLOR(c) (gl_FragColor = c)\n"
"precision highp float;\n"
"precision highp int;\n";
const char *shaderDecl300es =
"#version 300 es\n"
"#define VSIN(index) layout(location = index) in\n"
"#define VSOUT out\n"
"#define FSIN in\n"
"#define FRAGCOLOR(c) (fragColor = c)\n"
"precision highp float;\n"
"precision highp int;\n";

const char *shaderDecl310es =
"#version 310 es\n"
"#define VSIN(index) layout(location = index) in\n"
"#define VSOUT out\n"
"#define FSIN in\n"
"#define FRAGCOLOR(c) (fragColor = c)\n"
"precision highp float;\n"
"precision highp int;\n";

const char *shaderDecl;

// this needs a define in the shaders as well!
//#define RW_GL_USE_UBOS

static GLuint vao;
#ifdef RW_GL_USE_UBOS
static GLuint ubo_state, ubo_scene, ubo_object;
#endif
static GLuint whitetex;
static UniformState uniformState;
static UniformScene uniformScene;
static UniformObject uniformObject;

#ifndef RW_GL_USE_UBOS
// State
int32 u_alphaRef;
int32 u_fogData;
int32 u_fogColor;
#ifdef LIBRW_VISIONOS
int32 u_fogMode;   // x: 1 = radial fog (VC_FOG_RADIAL), see header.vert DoFogV
#endif

// Scene
int32 u_proj;
#ifdef LIBRW_VISIONOS
int32 u_projMV, u_viewMV;   // multiview twins' per-view matrices (5.0b/1)
#endif
int32 u_view;

// Object
int32 u_world;
int32 u_ambLight;
int32 u_lightParams;
int32 u_lightPosition;
int32 u_lightDirection;
int32 u_lightColor;
#endif

int32 u_matColor;
int32 u_surfProps;

Shader *defaultShader, *defaultShader_noAT;
Shader *defaultShader_fullLight, *defaultShader_fullLight_noAT;

static bool32 stateDirty = 1;
static bool32 sceneDirty = 1;
static bool32 objectDirty = 1;

struct RwRasterStateCache {
	Raster *raster;
	Texture::Addressing addressingU;
	Texture::Addressing addressingV;
	Texture::FilterMode filter;
};

#define MAXNUMSTAGES 8

// cached RW render states
struct RwStateCache {
	bool32 vertexAlpha;
	uint32 alphaTestEnable;
	uint32 alphaFunc;
	bool32 textureAlpha;
	bool32 blendEnable;
	uint32 srcblend, destblend;
	uint32 zwrite;
	uint32 ztest;
	uint32 cullmode;
	uint32 stencilenable;
	uint32 stencilpass;
	uint32 stencilfail;
	uint32 stencilzfail;
	uint32 stencilfunc;
	uint32 stencilref;
	uint32 stencilmask;
	uint32 stencilwritemask;
	uint32 fogEnable;
	float32 fogStart;
	float32 fogEnd;

	// emulation of PS2 GS
	bool32 gsalpha;
	uint32 gsalpharef;

	RwRasterStateCache texstage[MAXNUMSTAGES];
};
static RwStateCache rwStateCache;

enum
{
	// actual gl states
	RWGL_BLEND,
	RWGL_SRCBLEND,
	RWGL_DESTBLEND,
	RWGL_DEPTHTEST,
	RWGL_DEPTHFUNC,
	RWGL_DEPTHMASK,
	RWGL_CULL,
	RWGL_CULLFACE,
	RWGL_STENCIL,
	RWGL_STENCILFUNC,
	RWGL_STENCILFAIL,
	RWGL_STENCILZFAIL,
	RWGL_STENCILPASS,
	RWGL_STENCILREF,
	RWGL_STENCILMASK,
	RWGL_STENCILWRITEMASK,

	// uniforms
	RWGL_ALPHAFUNC,
	RWGL_ALPHAREF,
	RWGL_FOG,
	RWGL_FOGSTART,
	RWGL_FOGEND,
	RWGL_FOGCOLOR,

	RWGL_NUM_STATES
};
static bool uniformStateDirty[RWGL_NUM_STATES];

struct GlState {
	bool32 blendEnable;
	uint32 srcblend, destblend;

	bool32 depthTest;
	uint32 depthFunc;

	uint32 depthMask;

	bool32 cullEnable;
	uint32 cullFace;

	bool32 stencilEnable;
	// glStencilFunc
	uint32 stencilFunc;
	uint32 stencilRef;
	uint32 stencilMask;
	// glStencilOp
	uint32 stencilPass;
	uint32 stencilFail;
	uint32 stencilZFail;
	// glStencilMask
	uint32 stencilWriteMask;
};
static GlState curGlState, oldGlState;

static int32 activeTexture;
static uint32 boundTexture[MAXNUMSTAGES];

static uint32 currentFramebuffer;

static uint32 blendMap[] = {
	GL_ZERO,	// actually invalid
	GL_ZERO,
	GL_ONE,
	GL_SRC_COLOR,
	GL_ONE_MINUS_SRC_COLOR,
	GL_SRC_ALPHA,
	GL_ONE_MINUS_SRC_ALPHA,
	GL_DST_ALPHA,
	GL_ONE_MINUS_DST_ALPHA,
	GL_DST_COLOR,
	GL_ONE_MINUS_DST_COLOR,
	GL_SRC_ALPHA_SATURATE,
};

static uint32 stencilOpMap[] = {
	GL_KEEP,	// actually invalid
	GL_KEEP,
	GL_ZERO,
	GL_REPLACE,
	GL_INCR,
	GL_DECR,
	GL_INVERT,
	GL_INCR_WRAP,
	GL_DECR_WRAP
};

static uint32 stencilFuncMap[] = {
	GL_NEVER,	// actually invalid
	GL_NEVER,
	GL_LESS,
	GL_EQUAL,
	GL_LEQUAL,
	GL_GREATER,
	GL_NOTEQUAL,
	GL_GEQUAL,
	GL_ALWAYS
};

static float maxAnisotropy;

/*
 * GL state cache
 */

void
setGlRenderState(uint32 state, uint32 value)
{
	switch(state){
	case RWGL_BLEND: curGlState.blendEnable = value; break;
	case RWGL_SRCBLEND: curGlState.srcblend = value; break;
	case RWGL_DESTBLEND: curGlState.destblend = value; break;
	case RWGL_DEPTHTEST: curGlState.depthTest = value; break;
	case RWGL_DEPTHFUNC: curGlState.depthFunc = value; break;
	case RWGL_DEPTHMASK: curGlState.depthMask = value; break;
	case RWGL_CULL: curGlState.cullEnable = value; break;
	case RWGL_CULLFACE: curGlState.cullFace = value; break;
	case RWGL_STENCIL: curGlState.stencilEnable = value; break;
	case RWGL_STENCILFUNC: curGlState.stencilFunc = value; break;
	case RWGL_STENCILFAIL: curGlState.stencilFail = value; break;
	case RWGL_STENCILZFAIL: curGlState.stencilZFail = value; break;
	case RWGL_STENCILPASS: curGlState.stencilPass = value; break;
	case RWGL_STENCILREF: curGlState.stencilRef = value; break;
	case RWGL_STENCILMASK: curGlState.stencilMask = value; break;
	case RWGL_STENCILWRITEMASK: curGlState.stencilWriteMask = value; break;
	}
}

#ifdef LIBRW_VISIONOS
// VC_DOUBLE_RENDER cost probe hooks (toggled from main.cpp's RenderScene hook):
// mode 3 forces GL_LESS during the 2nd RenderScene so identical-depth fragments
// all fail the test -- overridden in flushGlRenderState (the single glDepthFunc
// chokepoint) so librw's LEQUAL can't leak back through its state cache.
static bool vc_gForceDepthLess = false;
extern "C" void vc_set_force_depth_less(int on) { vc_gForceDepthLess = on != 0; }
// mode 2: depth-only clear of the current (redirected) FBO, mask-safe, no RW
// detour (RwCameraClear mid-frame produced a colour artefact).
extern "C" void vc_clear_depth(void)
{
	glDepthMask(GL_TRUE);
	glClear(GL_DEPTH_BUFFER_BIT);
	uint32 mask = rwStateCache.zwrite ? GL_TRUE : GL_FALSE;
	glDepthMask(mask);
	oldGlState.depthMask = mask; // keep the low-level cache in sync
}
#endif

void
flushGlRenderState(void)
{
	if(oldGlState.blendEnable != curGlState.blendEnable){
		oldGlState.blendEnable = curGlState.blendEnable;
		(oldGlState.blendEnable ? glEnable : glDisable)(GL_BLEND);
	}

	if(oldGlState.srcblend != curGlState.srcblend ||
	   oldGlState.destblend != curGlState.destblend){
		oldGlState.srcblend = curGlState.srcblend;
		oldGlState.destblend = curGlState.destblend;
		glBlendFunc(oldGlState.srcblend, oldGlState.destblend);
	}

	if(oldGlState.depthTest != curGlState.depthTest){
		oldGlState.depthTest = curGlState.depthTest;
		(oldGlState.depthTest ? glEnable : glDisable)(GL_DEPTH_TEST);
	}
#ifdef LIBRW_VISIONOS
	{
		uint32 wantDepthFunc = vc_gForceDepthLess ? (uint32)GL_LESS : curGlState.depthFunc;
		if(oldGlState.depthFunc != wantDepthFunc){
			oldGlState.depthFunc = wantDepthFunc;
			glDepthFunc(oldGlState.depthFunc);
		}
	}
#else
	if(oldGlState.depthFunc != curGlState.depthFunc){
		oldGlState.depthFunc = curGlState.depthFunc;
		glDepthFunc(oldGlState.depthFunc);
	}
#endif
	if(oldGlState.depthMask != curGlState.depthMask){
		oldGlState.depthMask = curGlState.depthMask;
		glDepthMask(oldGlState.depthMask);
	}

	if(oldGlState.stencilEnable != curGlState.stencilEnable){
		oldGlState.stencilEnable = curGlState.stencilEnable;
		(oldGlState.stencilEnable ? glEnable : glDisable)(GL_STENCIL_TEST);
	}
	if(oldGlState.stencilFunc != curGlState.stencilFunc ||
	   oldGlState.stencilRef != curGlState.stencilRef ||
	   oldGlState.stencilMask != curGlState.stencilMask){
		oldGlState.stencilFunc = curGlState.stencilFunc;
		oldGlState.stencilRef = curGlState.stencilRef;
		oldGlState.stencilMask = curGlState.stencilMask;
		glStencilFunc(oldGlState.stencilFunc, oldGlState.stencilRef, oldGlState.stencilMask);
	}
	if(oldGlState.stencilPass != curGlState.stencilPass ||
	   oldGlState.stencilFail != curGlState.stencilFail ||
	   oldGlState.stencilZFail != curGlState.stencilZFail){
		oldGlState.stencilPass = curGlState.stencilPass;
		oldGlState.stencilFail = curGlState.stencilFail;
		oldGlState.stencilZFail = curGlState.stencilZFail;
		glStencilOp(oldGlState.stencilFail, oldGlState.stencilZFail, oldGlState.stencilPass);
	}
	if(oldGlState.stencilWriteMask != curGlState.stencilWriteMask){
		oldGlState.stencilWriteMask = curGlState.stencilWriteMask;
		glStencilMask(oldGlState.stencilWriteMask);
	}

	if(oldGlState.cullEnable != curGlState.cullEnable){
		oldGlState.cullEnable = curGlState.cullEnable;
		(oldGlState.cullEnable ? glEnable : glDisable)(GL_CULL_FACE);
	}
	if(oldGlState.cullFace != curGlState.cullFace){
		oldGlState.cullFace = curGlState.cullFace;
		glCullFace(oldGlState.cullFace);
	}
}



void
setAlphaBlend(bool32 enable)
{
	if(rwStateCache.blendEnable != enable){
		rwStateCache.blendEnable = enable;
		setGlRenderState(RWGL_BLEND, enable);
	}
}

bool32
getAlphaBlend(void)
{
	return rwStateCache.blendEnable;
}

bool32 getAlphaTest(void) { return rwStateCache.alphaTestEnable; }

static void
setDepthTest(bool32 enable)
{
	if(rwStateCache.ztest != enable){
		rwStateCache.ztest = enable;
		if(rwStateCache.zwrite && !enable){
			// If we still want to write, enable but set mode to always
			setGlRenderState(RWGL_DEPTHTEST, true);
			setGlRenderState(RWGL_DEPTHFUNC, GL_ALWAYS);
		}else{
			setGlRenderState(RWGL_DEPTHTEST, rwStateCache.ztest);
			setGlRenderState(RWGL_DEPTHFUNC, GL_LEQUAL);
		}
	}
}

static void
setDepthWrite(bool32 enable)
{
	enable = enable ? GL_TRUE : GL_FALSE;
	if(rwStateCache.zwrite != enable){
		rwStateCache.zwrite = enable;
		if(enable && !rwStateCache.ztest){
			// Have to switch on ztest so writing can work
			setGlRenderState(RWGL_DEPTHTEST, true);
			setGlRenderState(RWGL_DEPTHFUNC, GL_ALWAYS);
		}
		setGlRenderState(RWGL_DEPTHMASK, rwStateCache.zwrite);
	}
}

static void
setAlphaTest(bool32 enable)
{
	uint32 shaderfunc;
	if(rwStateCache.alphaTestEnable != enable){
		rwStateCache.alphaTestEnable = enable;
		shaderfunc = rwStateCache.alphaTestEnable ? rwStateCache.alphaFunc : ALPHAALWAYS;
		if(alphaFunc != shaderfunc){
			alphaFunc = shaderfunc;
			uniformStateDirty[RWGL_ALPHAFUNC] = true;
			stateDirty = 1;
		}
	}
}

static void
setAlphaTestFunction(uint32 function)
{
	uint32 shaderfunc;
	if(rwStateCache.alphaFunc != function){
		rwStateCache.alphaFunc = function;
		shaderfunc = rwStateCache.alphaTestEnable ? rwStateCache.alphaFunc : ALPHAALWAYS;
		if(alphaFunc != shaderfunc){
			alphaFunc = shaderfunc;
			uniformStateDirty[RWGL_ALPHAFUNC] = true;
			stateDirty = 1;
		}
	}
}

static void
setVertexAlpha(bool32 enable)
{
	if(rwStateCache.vertexAlpha != enable){
		if(!rwStateCache.textureAlpha){
			setAlphaBlend(enable);
			setAlphaTest(enable);
		}
		rwStateCache.vertexAlpha = enable;
	}
}

static void
setActiveTexture(int32 n)
{
	if(activeTexture != n){
		activeTexture = n;
		glActiveTexture(GL_TEXTURE0+n);
	}
}

uint32
bindTexture(uint32 texid)
{
	uint32 prev = boundTexture[activeTexture];
	if(prev != texid){
#ifdef LIBRW_VISIONOS
		g_vcTexBinds++;   // real glBindTexture calls, for the per-draw overhead split
#endif
		boundTexture[activeTexture] = texid;
		glBindTexture(GL_TEXTURE_2D, texid);
	}
	return prev;
}

void
bindFramebuffer(uint32 fbo)
{
	if(currentFramebuffer != fbo){
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		currentFramebuffer = fbo;
	}
#ifdef LIBRW_VISIONOS
	// 5.0b/2: program selection follows the binding (see Shader::use).
	{
		unsigned int mv = vc_multiview_fbo();
		vcMultiviewBound = (mv != 0 && fbo == mv);
	}
#endif
}

#ifdef LIBRW_VISIONOS
// 5.0b/2 self-test (called from the vc-mv5 import test with its 2-view FBO):
// registering that FBO as the multiview FBO and binding it through the librw
// wrapper must make use() pick the twin; rebinding the previous FBO must fall
// back to the mono program. Restores registration, binding and program.
// Returns bit2 = pairs available, bit0 = twin selected, bit1 = mono restored.
extern "C" int
vc_mv_bind_switch_selftest(unsigned int mvFbo)
{
	int r = 0;
	if(vc_mv_pairs_mode() != 1 || defaultShader == nil || defaultShader->mv == nil)
		return r;
	r |= 4;
	uint32 prevFbo = currentFramebuffer;
	Shader *prevShader = currentShader;
	// The test's own use() calls must not show up in the per-second selection
	// counters (device 2026-09-25: it produced the one "mv=1" FAIL line).
	unsigned mono0 = g_vcUseMono, mv0 = g_vcUseMv;
	vc_multiview_fbo_set(mvFbo);
	bindFramebuffer(mvFbo);
	defaultShader->use();
	if(currentShader == defaultShader->mv) r |= 1;
	vc_multiview_fbo_set(0);
	bindFramebuffer(prevFbo);
	defaultShader->use();
	if(currentShader == defaultShader) r |= 2;
	if(prevShader) prevShader->use();
	g_vcUseMono = mono0; g_vcUseMv = mv0;
	return r;
}
#endif

static GLint filterConvMap_NoMIP[] = {
	0, GL_NEAREST, GL_LINEAR,
	   GL_NEAREST, GL_LINEAR,
	   GL_NEAREST, GL_LINEAR
};
static GLint filterConvMap_MIP[] = {
	0, GL_NEAREST, GL_LINEAR,
	   GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_NEAREST,
	   GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR
};

static GLint addressConvMap[] = {
	0, GL_REPEAT, GL_MIRRORED_REPEAT,
	GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER
};

#ifdef LIBRW_VISIONOS
static int vcPerfLog(void){ static int v=-1; if(v<0){ v = getenv("VC_PERF_LOG") ? 1 : 0; } return v; }   // diagnostics gate
static int vcMipmapEnabled(void){ static int v=-1; if(v<0){ const char*s=getenv("VC_MIPMAP"); v=s?atoi(s):1; } return v; }
static int vcAnisoLevel(void){ static int v=-1; if(v<0){ const char*s=getenv("VC_ANISO"); v=s?atoi(s):16; if(v<1)v=1; if(v>16)v=16; } return v; }
static int vcMipmapAlpha(void){ static int v=-1; if(v<0){ const char*s=getenv("VC_MIPMAP_ALPHA"); v=s?atoi(s):1; } return v; }
// A raster carries our generated mip chain if it has >1 level and was eligible:
// opaque, OR alpha-tested when the VC_MIPMAP_ALPHA experiment is on.
static bool vcRasterMipped(Gl3Raster *natras){ return vcMipmapEnabled() && natras->numLevels > 1 && (!natras->hasAlpha || vcMipmapAlpha()); }
// Opaque world textures that got a generated mip chain (gl3raster VC_MIPMAP path:
// numLevels>1, no alpha) still arrive with filter=LINEAR, which maps to GL_LINEAR
// even in filterConvMap_MIP -> the chain would never be sampled. Upgrade the MIN
// filter to the trilinear variant so the mips are actually read. Alpha-masked
// textures are excluded here too (Stufe 2).
static int vcUpgradeMipFilter(Gl3Raster *natras, int32 filter){
	if(vcRasterMipped(natras)){
		if(filter == Texture::LINEAR)  return Texture::LINEARMIPLINEAR;   // trilinear
		if(filter == Texture::NEAREST) return Texture::MIPNEAREST;        // nearest + mip
	}
	return filter;
}
#endif

static void
setFilterMode(uint32 stage, int32 filter, int32 maxAniso = 1)
{
	if(rwStateCache.texstage[stage].filter != (Texture::FilterMode)filter){
		rwStateCache.texstage[stage].filter = (Texture::FilterMode)filter;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, rwStateCache.texstage[stage].raster, nativeRasterOffset);
			int32 effAniso = maxAniso;
#ifdef LIBRW_VISIONOS
			// Opaque mipped world textures run through THIS path and would otherwise
			// have aniso reset to the incoming maxAniso (=1). Enforce VC_ANISO here.
			if(vcRasterMipped(natras) && vcAnisoLevel() > effAniso)
				effAniso = vcAnisoLevel();
#endif
			if(natras->filterMode != filter){
				setActiveTexture(stage);
				if(natras->autogenMipmap || natras->numLevels > 1){
#ifdef LIBRW_VISIONOS
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[vcUpgradeMipFilter(natras, filter)]);
#else
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[filter]);
#endif
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}else{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_NoMIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}
				natras->filterMode = filter;
#ifdef LIBRW_VISIONOS
				// Measurement (NPC-face bleed): one line per texture the first time its
				// filter is set (dedup'd by this filterMode!=filter guard). Tells us whether
				// ped/face textures have mips (numLevels>1 or autogen -> coarse mip bleeds
				// clothing colour), which min-filter, and the requested aniso. Env-gated,
				// bounded during a cutscene. VC_TEXLOG=1 to enable.
				{
					static int texlog = -1;
					if(texlog < 0){ const char *s = getenv("VC_TEXLOG"); texlog = s ? atoi(s) : 0; }
					if(texlog)
						printf("[vc-tex] %dx%d numLevels=%d autogen=%d filter=%d->min=%d aniso=%d(req %d) hasAlpha=%d comp=%d cap=%.0f\n",
							raster->width, raster->height, natras->numLevels,
							(int)natras->autogenMipmap, filter, vcUpgradeMipFilter(natras, filter),
							effAniso, maxAniso, (int)natras->hasAlpha, (int)natras->isCompressed, gl3Caps.maxAnisotropy);
				}
#endif
			}
			if(natras->maxAnisotropy != effAniso){
				setActiveTexture(stage);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, (float)effAniso);
				natras->maxAnisotropy = effAniso;
			}
		}
	}
}

static void
setAddressU(uint32 stage, int32 addressing)
{
	if(rwStateCache.texstage[stage].addressingU != (Texture::Addressing)addressing){
		rwStateCache.texstage[stage].addressingU = (Texture::Addressing)addressing;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			if(natras->addressU == addressing){
				setActiveTexture(stage);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, addressConvMap[addressing]);
				natras->addressU = addressing;
			}
		}
	}
}

static void
setAddressV(uint32 stage, int32 addressing)
{
	if(rwStateCache.texstage[stage].addressingV != (Texture::Addressing)addressing){
		rwStateCache.texstage[stage].addressingV = (Texture::Addressing)addressing;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, rwStateCache.texstage[stage].raster, nativeRasterOffset);
			if(natras->addressV == addressing){
				setActiveTexture(stage);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, addressConvMap[addressing]);
				natras->addressV = addressing;
			}
		}
	}
}

static void
setRasterStageOnly(uint32 stage, Raster *raster)
{
	bool32 alpha;
	if(raster != rwStateCache.texstage[stage].raster){
		rwStateCache.texstage[stage].raster = raster;
		setActiveTexture(stage);
		if(raster){
			assert(raster->platform == PLATFORM_GL3);
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			bindTexture(natras->texid);

			rwStateCache.texstage[stage].filter = (rw::Texture::FilterMode)natras->filterMode;
			rwStateCache.texstage[stage].addressingU = (rw::Texture::Addressing)natras->addressU;
			rwStateCache.texstage[stage].addressingV = (rw::Texture::Addressing)natras->addressV;

			alpha = natras->hasAlpha;
		}else{
			bindTexture(whitetex);
			alpha = 0;
		}

		if(stage == 0){
			if(alpha != rwStateCache.textureAlpha){
				rwStateCache.textureAlpha = alpha;
				if(!rwStateCache.vertexAlpha){
					setAlphaBlend(alpha);
					setAlphaTest(alpha);
				}
			}
		}
	}
}

static void
setRasterStage(uint32 stage, Raster *raster)
{
	bool32 alpha;
	if(raster != rwStateCache.texstage[stage].raster){
		rwStateCache.texstage[stage].raster = raster;
		setActiveTexture(stage);
		if(raster){
			assert(raster->platform == PLATFORM_GL3);
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			bindTexture(natras->texid);
			uint32 filter = rwStateCache.texstage[stage].filter;
			uint32 addrU = rwStateCache.texstage[stage].addressingU;
			uint32 addrV = rwStateCache.texstage[stage].addressingV;
			if(natras->filterMode != filter){
				if(natras->autogenMipmap || natras->numLevels > 1){
#ifdef LIBRW_VISIONOS
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[vcUpgradeMipFilter(natras, filter)]);
#else
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[filter]);
#endif
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}else{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_NoMIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}
				natras->filterMode = filter;
			}
			if(natras->addressU != addrU){
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, addressConvMap[addrU]);
				natras->addressU = addrU;
			}
			if(natras->addressV != addrV){
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, addressConvMap[addrV]);
				natras->addressV = addrV;
			}
			alpha = natras->hasAlpha;
		}else{
			bindTexture(whitetex);
			alpha = 0;
		}

		if(stage == 0){
			if(alpha != rwStateCache.textureAlpha){
				rwStateCache.textureAlpha = alpha;
				if(!rwStateCache.vertexAlpha){
					setAlphaBlend(alpha);
					setAlphaTest(alpha);
				}
			}
		}
	}
}

void
evictRaster(Raster *raster)
{
	int i;
	for(i = 0; i < MAXNUMSTAGES; i++){
		//assert(rwStateCache.texstage[i].raster != raster);
		if(rwStateCache.texstage[i].raster != raster)
			continue;
		setRasterStage(i, nil);
	}
}

void
setTexture(int32 stage, Texture *tex)
{
	if(tex == nil || tex->raster == nil){
		setRasterStage(stage, nil);
		return;
	}
	setRasterStageOnly(stage, tex->raster);
	setFilterMode(stage, tex->getFilter(), tex->getMaxAnisotropy());
	setAddressU(stage, tex->getAddressU());
	setAddressV(stage, tex->getAddressV());
}

static void
setRenderState(int32 state, void *pvalue)
{
	uint32 value = (uint32)(uintptr)pvalue;
	switch(state){
	case TEXTURERASTER:
		setRasterStage(0, (Raster*)pvalue);
		break;
	case TEXTUREADDRESS:
		setAddressU(0, value);
		setAddressV(0, value);
		break;
	case TEXTUREADDRESSU:
		setAddressU(0, value);
		break;
	case TEXTUREADDRESSV:
		setAddressV(0, value);
		break;
	case TEXTUREFILTER:
		setFilterMode(0, value);
		break;
	case VERTEXALPHA:
		setVertexAlpha(value);
		break;
	case SRCBLEND:
		if(rwStateCache.srcblend != value){
			rwStateCache.srcblend = value;
			setGlRenderState(RWGL_SRCBLEND, blendMap[rwStateCache.srcblend]);
		}
		break;
	case DESTBLEND:
		if(rwStateCache.destblend != value){
			rwStateCache.destblend = value;
			setGlRenderState(RWGL_DESTBLEND, blendMap[rwStateCache.destblend]);
		}
		break;
	case ZTESTENABLE:
		setDepthTest(value);
		break;
	case ZWRITEENABLE:
		setDepthWrite(value);
		break;
	case FOGENABLE:
		if(rwStateCache.fogEnable != value){
			rwStateCache.fogEnable = value;
			uniformStateDirty[RWGL_FOG] = true;
			stateDirty = 1;
		}
		break;
	case FOGCOLOR:
		// no cache check here...too lazy
		RGBA c;
		c.red = value;
		c.green = value>>8;
		c.blue = value>>16;
		c.alpha = value>>24;
		convColor(&uniformState.fogColor, &c);
		uniformStateDirty[RWGL_FOGCOLOR] = true;
		stateDirty = 1;
		break;
	case CULLMODE:
		if(rwStateCache.cullmode != value){
			rwStateCache.cullmode = value;
			if(rwStateCache.cullmode == CULLNONE)
				setGlRenderState(RWGL_CULL, false);
			else{
				setGlRenderState(RWGL_CULL, true);
				setGlRenderState(RWGL_CULLFACE, rwStateCache.cullmode == CULLBACK ? GL_BACK : GL_FRONT);
			}
		}
		break;

	case STENCILENABLE:
		if(rwStateCache.stencilenable != value){
			rwStateCache.stencilenable = value;
			setGlRenderState(RWGL_STENCIL, value);
		}
		break;
	case STENCILFAIL:
		if(rwStateCache.stencilfail != value){
			rwStateCache.stencilfail = value;
			setGlRenderState(RWGL_STENCILFAIL, stencilOpMap[value]);
		}
		break;
	case STENCILZFAIL:
		if(rwStateCache.stencilzfail != value){
			rwStateCache.stencilzfail = value;
			setGlRenderState(RWGL_STENCILZFAIL, stencilOpMap[value]);
		}
		break;
	case STENCILPASS:
		if(rwStateCache.stencilpass != value){
			rwStateCache.stencilpass = value;
			setGlRenderState(RWGL_STENCILPASS, stencilOpMap[value]);
		}
		break;
	case STENCILFUNCTION:
		if(rwStateCache.stencilfunc != value){
			rwStateCache.stencilfunc = value;
			setGlRenderState(RWGL_STENCILFUNC, stencilFuncMap[value]);
		}
		break;
	case STENCILFUNCTIONREF:
		if(rwStateCache.stencilref != value){
			rwStateCache.stencilref = value;
			setGlRenderState(RWGL_STENCILREF, value);
		}
		break;
	case STENCILFUNCTIONMASK:
		if(rwStateCache.stencilmask != value){
			rwStateCache.stencilmask = value;
			setGlRenderState(RWGL_STENCILMASK, value);
		}
		break;
	case STENCILFUNCTIONWRITEMASK:
		if(rwStateCache.stencilwritemask != value){
			rwStateCache.stencilwritemask = value;
			setGlRenderState(RWGL_STENCILWRITEMASK, value);
		}
		break;

	case ALPHATESTFUNC:
		setAlphaTestFunction(value);
		break;
	case ALPHATESTREF:
		if(alphaRef != value/255.0f){
			alphaRef = value/255.0f;
			uniformStateDirty[RWGL_ALPHAREF] = true;
			stateDirty = 1;
		}
		break;
	case GSALPHATEST:
		rwStateCache.gsalpha = value;
		break;
	case GSALPHATESTREF:
		rwStateCache.gsalpharef = value;
	}
}

static void*
getRenderState(int32 state)
{
	uint32 val;
	RGBA rgba;
	switch(state){
	case TEXTURERASTER:
		return rwStateCache.texstage[0].raster;
	case TEXTUREADDRESS:
		if(rwStateCache.texstage[0].addressingU == rwStateCache.texstage[0].addressingV)
			val = rwStateCache.texstage[0].addressingU;
		else
			val = 0;	// invalid
		break;
	case TEXTUREADDRESSU:
		val = rwStateCache.texstage[0].addressingU;
		break;
	case TEXTUREADDRESSV:
		val = rwStateCache.texstage[0].addressingV;
		break;
	case TEXTUREFILTER:
		val = rwStateCache.texstage[0].filter;
		break;

	case VERTEXALPHA:
		val = rwStateCache.vertexAlpha;
		break;
	case SRCBLEND:
		val = rwStateCache.srcblend;
		break;
	case DESTBLEND:
		val = rwStateCache.destblend;
		break;
	case ZTESTENABLE:
		val = rwStateCache.ztest;
		break;
	case ZWRITEENABLE:
		val = rwStateCache.zwrite;
		break;
	case FOGENABLE:
		val = rwStateCache.fogEnable;
		break;
	case FOGCOLOR:
		convColor(&rgba, &uniformState.fogColor);
		val = RWRGBAINT(rgba.red, rgba.green, rgba.blue, rgba.alpha);
		break;
	case CULLMODE:
		val = rwStateCache.cullmode;
		break;

	case STENCILENABLE:
		val = rwStateCache.stencilenable;
		break;
	case STENCILFAIL:
		val = rwStateCache.stencilfail;
		break;
	case STENCILZFAIL:
		val = rwStateCache.stencilzfail;
		break;
	case STENCILPASS:
		val = rwStateCache.stencilpass;
		break;
	case STENCILFUNCTION:
		val = rwStateCache.stencilfunc;
		break;
	case STENCILFUNCTIONREF:
		val = rwStateCache.stencilref;
		break;
	case STENCILFUNCTIONMASK:
		val = rwStateCache.stencilmask;
		break;
	case STENCILFUNCTIONWRITEMASK:
		val = rwStateCache.stencilwritemask;
		break;

	case ALPHATESTFUNC:
		val = rwStateCache.alphaFunc;
		break;
	case ALPHATESTREF:
		val = (uint32)(alphaRef*255.0f);
		break;
	case GSALPHATEST:
		val = rwStateCache.gsalpha;
		break;
	case GSALPHATESTREF:
		val = rwStateCache.gsalpharef;
		break;
	default:
		val = 0;
	}
	return (void*)(uintptr)val;
}

static void
resetRenderState(void)
{	
	rwStateCache.alphaFunc = ALPHAGREATEREQUAL;
	alphaFunc = 0;
	alphaRef = 10.0f/255.0f;
	uniformState.fogDisable = 1.0f;
	uniformState.fogStart = 0.0f;
	uniformState.fogEnd = 0.0f;
	uniformState.fogRange = 0.0f;
	uniformState.fogColor = { 1.0f, 1.0f, 1.0f, 1.0f };
	rwStateCache.gsalpha = 0;
	rwStateCache.gsalpharef = 128;
	stateDirty = 1;

	rwStateCache.vertexAlpha = 0;
	rwStateCache.textureAlpha = 0;
	rwStateCache.alphaTestEnable = 0;

	memset(&oldGlState, 0xFE, sizeof(oldGlState));

	rwStateCache.blendEnable = 0;
	setGlRenderState(RWGL_BLEND, false);
	rwStateCache.srcblend = BLENDSRCALPHA;
	rwStateCache.destblend = BLENDINVSRCALPHA;
	setGlRenderState(RWGL_SRCBLEND, blendMap[rwStateCache.srcblend]);
	setGlRenderState(RWGL_DESTBLEND, blendMap[rwStateCache.destblend]);

	rwStateCache.zwrite = GL_TRUE;
	setGlRenderState(RWGL_DEPTHMASK, rwStateCache.zwrite);

	rwStateCache.ztest = 1;
	setGlRenderState(RWGL_DEPTHTEST, true);
	setGlRenderState(RWGL_DEPTHFUNC, GL_LEQUAL);

	rwStateCache.cullmode = CULLNONE;
	setGlRenderState(RWGL_CULL, false);
	setGlRenderState(RWGL_CULLFACE, GL_BACK);

	rwStateCache.stencilenable = 0;
	setGlRenderState(RWGL_STENCIL, GL_FALSE);
	rwStateCache.stencilfail = STENCILKEEP;
	setGlRenderState(RWGL_STENCILFAIL, GL_KEEP);
	rwStateCache.stencilzfail = STENCILKEEP;
	setGlRenderState(RWGL_STENCILZFAIL, GL_KEEP);
	rwStateCache.stencilpass = STENCILKEEP;
	setGlRenderState(RWGL_STENCILPASS, GL_KEEP);
	rwStateCache.stencilfunc = STENCILALWAYS;
	setGlRenderState(RWGL_STENCILFUNC, GL_ALWAYS);
	rwStateCache.stencilref = 0;
	setGlRenderState(RWGL_STENCILREF, 0);
	rwStateCache.stencilmask = 0xFFFFFFFF;
	setGlRenderState(RWGL_STENCILMASK, 0xFFFFFFFF);
	rwStateCache.stencilwritemask = 0xFFFFFFFF;
	setGlRenderState(RWGL_STENCILWRITEMASK, 0xFFFFFFFF);

	activeTexture = -1;
	for(int i = 0; i < MAXNUMSTAGES; i++){
		setActiveTexture(i);
		bindTexture(whitetex);
	}
	setActiveTexture(0);
}

void
setWorldMatrix(Matrix *mat)
{
	convMatrix(&uniformObject.world, mat);
	setUniform(u_world, &uniformObject.world);
	objectDirty = 1;
}

int32
setLights(WorldLights *lightData)
{
	int i, n;
	Light *l;
	int32 bits;

	uniformObject.ambLight = lightData->ambient;

	bits = 0;

	if(lightData->numAmbients)
		bits |= VSLIGHT_AMBIENT;

	n = 0;
	for(i = 0; i < lightData->numDirectionals && i < 8; i++){
		l = lightData->directionals[i];
		uniformObject.lightParams[n].type = 1.0f;
		uniformObject.lightColor[n] = l->color;
		memcpy(&uniformObject.lightDirection[n], &l->getFrame()->getLTM()->at, sizeof(V3d));
		bits |= VSLIGHT_DIRECT;
		n++;
		if(n >= MAX_LIGHTS)
			goto out;
	}

	for(i = 0; i < lightData->numLocals; i++){
		Light *l = lightData->locals[i];

		switch(l->getType()){
		case Light::POINT:
			uniformObject.lightParams[n].type = 2.0f;
			uniformObject.lightParams[n].radius = l->radius;
			uniformObject.lightColor[n] = l->color;
			memcpy(&uniformObject.lightPosition[n], &l->getFrame()->getLTM()->pos, sizeof(V3d));
			bits |= VSLIGHT_POINT;
			n++;
			if(n >= MAX_LIGHTS)
				goto out;
			break;
		case Light::SPOT:
		case Light::SOFTSPOT:
			uniformObject.lightParams[n].type = 3.0f;
			uniformObject.lightParams[n].minusCosAngle = l->minusCosAngle;
			uniformObject.lightParams[n].radius = l->radius;
			uniformObject.lightColor[n] = l->color;
			memcpy(&uniformObject.lightPosition[n], &l->getFrame()->getLTM()->pos, sizeof(V3d));
			memcpy(&uniformObject.lightDirection[n], &l->getFrame()->getLTM()->at, sizeof(V3d));
			// lower bound of falloff
			if(l->getType() == Light::SOFTSPOT)
				uniformObject.lightParams[n].hardSpot = 0.0f;
			else
				uniformObject.lightParams[n].hardSpot = 1.0f;
			bits |= VSLIGHT_SPOT;
			n++;
			if(n >= MAX_LIGHTS)
				goto out;
			break;
		}
	}

	uniformObject.lightParams[n].type = 0.0f;

	setUniform(u_ambLight, &uniformObject.ambLight);
	setUniform(u_lightParams, uniformObject.lightParams);
	setUniform(u_lightPosition, uniformObject.lightPosition);
	setUniform(u_lightDirection, uniformObject.lightDirection);
	setUniform(u_lightColor, uniformObject.lightColor);
out:
	objectDirty = 1;
	return bits;
}

void
setProjectionMatrix(float32 *mat)
{
	memcpy(&uniformScene.proj, mat, 64);
	setUniform(u_proj, uniformScene.proj);
	sceneDirty = 1;
}

void
setViewMatrix(float32 *mat)
{
	memcpy(&uniformScene.view, mat, 64);
	setUniform(u_view, uniformScene.view);
	sceneDirty = 1;
}

Shader *lastShaderUploaded;

void
setMaterial(const RGBA &color, const SurfaceProperties &surfaceprops, float extraSurfProp)
{
	rw::RGBAf col;
	convColor(&col, &color);
	setUniform(u_matColor, &col);

	float surfProps[4];
	surfProps[0] = surfaceprops.ambient;
	surfProps[1] = surfaceprops.specular;
	surfProps[2] = surfaceprops.diffuse;
	surfProps[3] = extraSurfProp;
	setUniform(u_surfProps, surfProps);
}

void
flushCache(void)
{
	flushGlRenderState();

#ifndef RW_GL_USE_UBOS

	// what's this doing here??
	uniformState.fogDisable = rwStateCache.fogEnable ? 0.0f : 1.0f;
	uniformState.fogStart = rwStateCache.fogStart;
	uniformState.fogEnd = rwStateCache.fogEnd;
	uniformState.fogRange = 1.0f/(rwStateCache.fogStart - rwStateCache.fogEnd);

	if(uniformStateDirty[RWGL_ALPHAFUNC] || uniformStateDirty[RWGL_ALPHAREF]){
		float alphaTest[4];
		switch(alphaFunc){
		case ALPHAALWAYS:
		default:
			alphaTest[0] = -1000.0f;
			alphaTest[1] = 1000.0f;
			break;
		case ALPHAGREATEREQUAL:
			alphaTest[0] = alphaRef;
			alphaTest[1] = 1000.0f;
			break;
		case ALPHALESS:
			alphaTest[0] = -1000.0f;
			alphaTest[1] = alphaRef;
			break;
		}
		setUniform(u_alphaRef, alphaTest);
		uniformStateDirty[RWGL_ALPHAFUNC] = false;
		uniformStateDirty[RWGL_ALPHAREF] = false;
	}

	if(uniformStateDirty[RWGL_FOG] ||
	   uniformStateDirty[RWGL_FOGSTART] ||
	   uniformStateDirty[RWGL_FOGEND]){
		float fogData[4] = {
			uniformState.fogStart,
			uniformState.fogEnd,
			uniformState.fogRange,
			uniformState.fogDisable
		};
		setUniform(u_fogData, fogData);
		uniformStateDirty[RWGL_FOG] = false;
		uniformStateDirty[RWGL_FOGSTART] = false;
		uniformStateDirty[RWGL_FOGEND] = false;
	}

	if(uniformStateDirty[RWGL_FOGCOLOR]){
		setUniform(u_fogColor, &uniformState.fogColor);
		uniformStateDirty[RWGL_FOGCOLOR] = false;
	}

#else
	if(objectDirty){
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_object);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), &uniformObject, GL_STREAM_DRAW);
		objectDirty = 0;
	}
	if(sceneDirty){
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_scene);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), &uniformScene, GL_STREAM_DRAW);
		sceneDirty = 0;
	}
	if(stateDirty){
		switch(alphaFunc){
		case ALPHAALWAYS:
		default:
			uniformState.alphaRefLow = -1000.0f;
			uniformState.alphaRefHigh = 1000.0f;
			break;
		case ALPHAGREATEREQUAL:
			uniformState.alphaRefLow = alphaRef;
			uniformState.alphaRefHigh = 1000.0f;
			break;
		case ALPHALESS:
			uniformState.alphaRefLow = -1000.0f;
			uniformState.alphaRefHigh = alphaRef;
			break;
		}
		uniformState.fogDisable = rwStateCache.fogEnable ? 0.0f : 1.0f;
		uniformState.fogStart = rwStateCache.fogStart;
		uniformState.fogEnd = rwStateCache.fogEnd;
		uniformState.fogRange = 1.0f/(rwStateCache.fogStart - rwStateCache.fogEnd);
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_state);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), &uniformState, GL_STREAM_DRAW);
		stateDirty = 0;
	}
#endif
	flushUniforms();
}

static void
setFrameBuffer(Camera *cam)
{
	Raster *fbuf = cam->frameBuffer->parent;
	Raster *zbuf = cam->zBuffer->parent;
	assert(fbuf);

	Gl3Raster *natfb = PLUGINOFFSET(Gl3Raster, fbuf, nativeRasterOffset);
	Gl3Raster *natzb = PLUGINOFFSET(Gl3Raster, zbuf, nativeRasterOffset);
	assert(fbuf->type == Raster::CAMERA || fbuf->type == Raster::CAMERATEXTURE);

	uint32 fbo = natfb->fbo;
#ifdef LIBRW_VISIONOS
	bool vcRedirected = false;
	// The GLES context is surfaceless, so the "default framebuffer" (fbo 0) does
	// not exist. When the main camera (a CAMERA raster, fbo 0) would bind it,
	// redirect to the platform's external EGLImage-backed FBO instead. Gated on
	// an explicit named hook so this can't fire for an unrelated fbo==0 and so
	// the intent is visible. Render-to-texture cameras keep their own fbo.
	if(fbo == 0 && vc_use_external_framebuffer()){
		fbo = vc_external_framebuffer();
		vcRedirected = true;
		static bool vcLoggedRedirect = false;
		if(!vcLoggedRedirect){
			vcLoggedRedirect = true;
			if(vcPerfLog()) printf("[vc-fb] redirecting camera default framebuffer (0) -> external FBO %u\n", fbo);
		}
	}
#endif

	// Have to make sure depth buffer is attached to FB's fbo
	bindFramebuffer(fbo);
	if(zbuf){
#ifdef LIBRW_VISIONOS
		// DIAGNOSE: log ONLY when the framebuffer or the zbuffer changes vs the
		// previous call -- silent while stable, one line at each switch. Capture
		// the fast-path flag BEFORE the attach (it sets fboMate = zbuf), and read
		// the ACTUALLY bound depth OBJECT_NAME AFTER, so "should" (zTexid) vs "is"
		// (boundDepthName) is visible -- decisive when the fast path attaches
		// nothing.
		bool vcLogChange = false;
		bool vcFast = (natfb->fboMate == zbuf);
		{
			static uint32 vcPrevFbo = 0xFFFFFFFFu, vcPrevZ = 0xFFFFFFFFu;
			if(fbo != vcPrevFbo || natzb->texid != vcPrevZ){
				vcPrevFbo = fbo;
				vcPrevZ = natzb->texid;
				vcLogChange = true;
			}
		}
#endif
		if(natfb->fboMate == zbuf){
			// all good
			assert(natzb->fboMate == fbuf);
		}else{
			if(natzb->fboMate){
				// have to detatch from fbo first!
				Gl3Raster *oldfb = PLUGINOFFSET(Gl3Raster, natzb->fboMate, nativeRasterOffset);
				if(oldfb->fbo){
					bindFramebuffer(oldfb->fbo);
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
					bindFramebuffer(fbo);
				}
				oldfb->fboMate = nil;
			}
			natfb->fboMate = zbuf;
			natzb->fboMate = fbuf;
			if(fbo){
#ifdef LIBRW_VISIONOS
				if(vcRedirected){
					// Both back buffers share this one CAMERA raster, so librw's
					// per-raster fboMate bookkeeping only ever tracks one FBO. Attach
					// the shared depth renderbuffer to BOTH external FBOs here (once);
					// librw's "all good" fast path is then correct for either buffer.
					vc_attach_depth_renderbuffer(natzb->texid);
				}else
#endif
				if(gl3Caps.gles)
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, natzb->texid);
				else
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, natzb->texid, 0);
			}
		}
#ifdef LIBRW_VISIONOS
		// After the (possibly skipped) attach, read what is REALLY bound as depth
		// on the now-current fbo. boundDepthName == zTexid -> "is" matches "should";
		// a mismatch (esp. on the fast path, which attaches nothing) is the bug.
		if(vcLogChange && vcPerfLog()){
			GLint boundDepthName = 0;
			glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &boundDepthName);
			printf("[vc-fb] setFrameBuffer CHANGE fbo=%u redirected=%d fastPath=%d gles=%d zTexid=%u zdims=%dx%d boundDepthOBJECT_NAME=%d\n",
			       fbo, vcRedirected ? 1 : 0, vcFast ? 1 : 0, gl3Caps.gles ? 1 : 0,
			       natzb->texid, zbuf->width, zbuf->height, (int)boundDepthName);
		}
#endif
	}else{
		// remove z-buffer
		if(natfb->fboMate && fbo)
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
		natfb->fboMate = nil;
	}
}

static Rect
getFramebufferRect(Raster *frameBuffer)
{
	Rect r;
	Raster *fb = frameBuffer->parent;
	if(fb->type == Raster::CAMERA){
#if defined(LIBRW_VISIONOS)
		// TODO(visionos): framebuffer size comes from the external ANGLE/Metal target; use the stub mode for now.
		r.w = glGlobals.modes[glGlobals.currentMode].mode.width;
		r.h = glGlobals.modes[glGlobals.currentMode].mode.height;
#elif defined(LIBRW_SDL2)
		SDL_GetWindowSize(glGlobals.window, &r.w, &r.h);
#else
		glfwGetFramebufferSize(glGlobals.window, &r.w, &r.h);
#endif
	}else{
		r.w = fb->width;
		r.h = fb->height;
	}
	r.x = 0;
	r.y = 0;

	// Got a subraster
	if(frameBuffer != fb){
		r.x = frameBuffer->offsetX;
		// GL y offset is from bottom
		r.y = r.h - frameBuffer->height - frameBuffer->offsetY;
		r.w = frameBuffer->width;
		r.h = frameBuffer->height;
	}

	return r;
}

static void
setViewport(Raster *frameBuffer)
{
	Rect r = getFramebufferRect(frameBuffer);
	if(r.w != glGlobals.presentWidth || r.h != glGlobals.presentHeight ||
	   r.x != glGlobals.presentOffX || r.y != glGlobals.presentOffY){
		glViewport(r.x, r.y, r.w, r.h);
		glGlobals.presentWidth = r.w;
		glGlobals.presentHeight = r.h;
		glGlobals.presentOffX = r.x;
		glGlobals.presentOffY = r.y;
	}
}

#ifdef LIBRW_VISIONOS
// Is `cam` the MAIN (screen) camera -- the one whose pixels we publish to the
// visionOS compositor -- rather than a render-to-texture camera (shadow maps,
// water reflections, motion blur)? This uses the SAME test as the framebuffer
// redirect in setFrameBuffer(): the main camera targets a CAMERA-type raster
// whose native fbo is 0 (the surfaceless "default" framebuffer), which we
// redirect to the external EGLImage FBO. RTT cameras target CAMERATEXTURE
// rasters with their own non-zero fbo and never match. Named explicitly (not a
// side effect) so the head-pose injection can be scoped to exactly this camera:
// injecting the head offset into the RTT cameras would drag shadows, water and
// blur with the head -- and, once stereo renders two passes, differently per
// eye (shadows swimming between the eyes).
// The main camera's actually-uploaded view/proj this frame (librw convention,
// column-major). Stashed in beginUpdate so the stereo eye passes -- which run
// OUTSIDE beginUpdate, at the RenderScene hook -- can start from them and add a
// per-eye offset. Written only for the main camera.
static float vcMainView[16];
static float vcMainProj[16];
// The main camera's near/far (GAME UNITS) this frame, stashed alongside so the
// eye passes can build a librw-depth projection (the compositor projection is
// reverse-Z with near/far in METRES -- unusable for the game's depth buffer).
static float vcMainNear = 1.0f;
static float vcMainFar  = 1000.0f;
// The per-eye VIEW + PROJECTION (librw) the last eye pass uploaded (see vc_get_eye_view).
static float vcEyeView[16];
static float vcEyeProj[16];
static int   vcEyeViewValid = 0;
#define g_vcEyeView vcEyeView
#define g_vcEyeProj vcEyeProj
#define g_vcEyeViewValid vcEyeViewValid
static float g_vcEyeViewMV[32];       // one-pass: both eyes' views (diagnostics, vc_get_eye_view_mv)
static int   g_vcEyeViewMVValid = 0;
// Stereo sky clear colour (0..1), set per frame from CTimeCycle by main.cpp. Default
// black = old behaviour (so nothing changes until the reVC side pushes a colour).
static float g_vcSkyClear[3] = { 0.0f, 0.0f, 0.0f };
extern "C" void vc_set_stereo_sky_clear(float r, float g, float b)
{
	g_vcSkyClear[0] = r; g_vcSkyClear[1] = g; g_vcSkyClear[2] = b;
}

static bool
vcIsMainCamera(Camera *cam)
{
	if(!vc_use_external_framebuffer())
		return false;
	if(cam->frameBuffer == nil)
		return false;
	Raster *fbuf = cam->frameBuffer->parent;
	if(fbuf == nil)
		return false;
	Gl3Raster *natfb = PLUGINOFFSET(Gl3Raster, fbuf, nativeRasterOffset);
	return fbuf->type == Raster::CAMERA && natfb->fbo == 0;
}
#endif

static void
beginUpdate(Camera *cam)
{
	float view[16], proj[16];
	// View Matrix
	Matrix inv;
	Matrix::invert(&inv, cam->getFrame()->getLTM());
	// Since we're looking into positive Z,
	// flip X to ge a left handed view space.
	view[0]  = -inv.right.x;
	view[1]  =  inv.right.y;
	view[2]  =  inv.right.z;
	view[3]  =  0.0f;
	view[4]  = -inv.up.x;
	view[5]  =  inv.up.y;
	view[6]  =  inv.up.z;
	view[7]  =  0.0f;
	view[8]  =  -inv.at.x;
	view[9]  =   inv.at.y;
	view[10] =  inv.at.z;
	view[11] =  0.0f;
	view[12] = -inv.pos.x;
	view[13] =  inv.pos.y;
	view[14] =  inv.pos.z;
	view[15] =  1.0f;
	memcpy(&cam->devView, &view, sizeof(RawMatrix));
	setViewMatrix(view);

	// Projection Matrix
	float32 invwx = 1.0f/cam->viewWindow.x;
	float32 invwy = 1.0f/cam->viewWindow.y;
	float32 invz = 1.0f/(cam->farPlane-cam->nearPlane);

	proj[0] = invwx;
	proj[1] = 0.0f;
	proj[2] = 0.0f;
	proj[3] = 0.0f;

	proj[4] = 0.0f;
	proj[5] = invwy;
	proj[6] = 0.0f;
	proj[7] = 0.0f;

	proj[8] = cam->viewOffset.x*invwx;
	proj[9] = cam->viewOffset.y*invwy;
	proj[12] = -proj[8];
	proj[13] = -proj[9];
	if(cam->projection == Camera::PERSPECTIVE){
		proj[10] = (cam->farPlane+cam->nearPlane)*invz;
		proj[11] = 1.0f;

		proj[14] = -2.0f*cam->nearPlane*cam->farPlane*invz;
		proj[15] = 0.0f;
	}else{
		proj[10] = 2.0f*invz;
		proj[11] = 0.0f;

		proj[14] = -(cam->farPlane+cam->nearPlane)*invz;
		proj[15] = 1.0f;
	}
	memcpy(&cam->devProj, &proj, sizeof(RawMatrix));
	setProjectionMatrix(proj);

#ifdef LIBRW_VISIONOS
	// Stereo injection point: both matrices are now fully formed (librw
	// convention) and just uploaded. VC_MATRIX_TEST drives the C-naht seam --
	// identity feeds these same matrices back (image unchanged), shift adds a
	// fixed view offset (image moves). Then, if the override is active, re-upload
	// the seam's matrices so gl3device shows THEM instead of its own. Only the
	// uniforms change; RwCamera's own frame/frustum (culling, LOD, water,
	// shadows) are untouched, so game logic and image can differ -- fine for a
	// small stereo offset, expected for the 0.5 m shift test.
	//
	// Scoped to the MAIN camera only: beginUpdate runs once per camera per frame
	// (screen camera + RTT cameras for shadows/water/blur). Injecting the head
	// offset into the RTT cameras would drag their results with the head, so we
	// gate on vcIsMainCamera(). The RTT cameras keep their own computed view/proj.
	{
		bool vcMain = vcIsMainCamera(cam);
		bool vcInjected = false;
		if(vcMain){
			// Stash the mono main matrices for the stereo eye passes. If the
			// override fires below, these are overwritten with the composed ones.
			memcpy(vcMainView, view, 16*sizeof(float));
			memcpy(vcMainProj, proj, 16*sizeof(float));
			vcMainNear = cam->nearPlane;
			vcMainFar  = cam->farPlane;
			int vcMT = vc_matrix_test_mode();
			if(vcMT != 0){
				float tv[16], tp[16];
				memcpy(tv, view, 16*sizeof(float));
				memcpy(tp, proj, 16*sizeof(float));
				if(vcMT == 2)
					tv[12] += 0.5f; // +0.5 m along view-space X (sign per librw's X-flip; test only needs a visible shift)
				vc_set_view_matrix(tv);
				vc_set_projection_matrix(tp);
				vc_set_matrix_override(1);
			}
			if(vc_matrix_override_active()){
				float ov[16], op[16];
				vc_get_view_matrix(ov);
				vc_get_projection_matrix(op);
				float vfinal[16];
				if(vc_view_compose_active()){
					// Head-pose offset: V_final = ov * view (column-major). Game view
					// (culling/LOD source) stays intact; the head only adds on top.
					for(int c = 0; c < 4; c++)
						for(int r = 0; r < 4; r++)
							vfinal[c*4+r] = ov[0*4+r]*view[c*4+0] + ov[1*4+r]*view[c*4+1]
							              + ov[2*4+r]*view[c*4+2] + ov[3*4+r]*view[c*4+3];
				}else{
					memcpy(vfinal, ov, 16*sizeof(float)); // replace (matrix self-test)
				}
				// Throttled diagnostic: what the compose actually consumes. compose=1
				// means V_final=ov*view (offset), 0 means replace (camera = ov only).
				// view t = game camera (should sit at Tommy); ov t = head offset t;
				// ov row0 (c0,c2) = yaw cos/sin; vfinal t = final camera position.
				// Isolation switch: VC_HEAD_KEEP_PROJ=1 keeps the game's OWN projection
				// (proj) and only overrides the view -- to tell a projection bug (pOut)
				// apart from a view/compose bug when the scene looks wrong at rest.
				static int vcKeepProj = -1;
				if(vcKeepProj < 0) vcKeepProj = getenv("VC_HEAD_KEEP_PROJ") ? 1 : 0;
				float *finalProj = vcKeepProj ? proj : op;

				static int vcHeadGlN = 0;
				if((vcHeadGlN++ % 120) == 0 && vcPerfLog()){
					Matrix *ltm = cam->getFrame()->getLTM();
					printf("[vc-head-gl] compose=%d keepProj=%d  ov_yaw[c0=%.3f c2=%.3f]  cam_up=(%.3f %.3f %.3f)  cam_at=(%.3f %.3f %.3f)\n",
					       vc_view_compose_active(), vcKeepProj,
					       ov[0], ov[8],
					       ltm->up.x, ltm->up.y, ltm->up.z,
					       ltm->at.x, ltm->at.y, ltm->at.z);
				}
				memcpy(&cam->devView, vfinal, sizeof(RawMatrix));
				memcpy(&cam->devProj, finalProj, sizeof(RawMatrix));
				setViewMatrix(vfinal);
				setProjectionMatrix(finalProj);
				// The composed matrices are what the eye passes must offset from.
				memcpy(vcMainView, vfinal, 16*sizeof(float));
				memcpy(vcMainProj, finalProj, 16*sizeof(float));
				vcInjected = true;
			}
		}

		// Prove the scoping once per session: count beginUpdate calls and how many
		// actually got the injection across one full frame (the span between two
		// consecutive main-camera renders). Expect injected == 1 of N.
		{
			static int vcCalls = 0, vcInjN = 0, vcMainSeen = 0;
			static bool vcLoggedScope = false;
			vcCalls++;
			if(vcInjected) vcInjN++;
			if(vcMain){
				if(++vcMainSeen >= 2 && !vcLoggedScope){
					vcLoggedScope = true;
					if(vcPerfLog()) printf("[vc-head-scope] beginUpdate calls/frame=%d injected/frame=%d (expect 1 of N)\n",
					       vcCalls, vcInjN);
				}
				vcCalls = 0; vcInjN = 0; // reset for the next inter-main interval
			}
		}
	}
#endif

	if(rwStateCache.fogStart != cam->fogPlane){
		rwStateCache.fogStart = cam->fogPlane;
		uniformStateDirty[RWGL_FOGSTART] = true;
		stateDirty = 1;
	}
	if(rwStateCache.fogEnd != cam->farPlane){
		rwStateCache.fogEnd = cam->farPlane;
		uniformStateDirty[RWGL_FOGEND] = true;
		stateDirty = 1;
	}

	setFrameBuffer(cam);

	setViewport(cam->frameBuffer);
}

#ifdef LIBRW_VISIONOS
// Phase 5.5 stereo eye pass. Called from main.cpp's RenderScene hook (inside the
// main camera's Begin/End) once per eye, BEFORE re-running RenderScene. Binds
// the eye's slice FBO (its own depth), clears, and uploads the stashed main
// matrices with a synthetic per-eye view-space X offset -- enough to prove the
// two slices differ. (Real per-eye compositor matrices need the Swift side and
// are the next step.) Deliberately bypasses the vc_external_framebuffer redirect
// so the cinema path and vcIsMainCamera() are untouched; vc_stereo_restore_main
// puts the cinema binding back afterwards.
extern "C" void
vc_stereo_eye_pass(int eye)
{
	// The prologue was OUTSIDE the brackets and held ~4.9 of 5.0 ms: with MSAA on,
	// vc_stereo_eye_fbo resolves the PREVIOUS eye here (glBlitFramebuffer), which cannot
	// start before that eye's rendering has finished on the GPU. Bracketed now so the
	// wait is attributed instead of hiding in a gap.
	// Book everything from here on under THIS eye (vcEyeTag in main.cpp is only set after
	// this function returns, so the boundary stages would all land in eye 0's bucket).
	vc_scene_set_eye(eye);
	vc_scene_begin(VC_SC_EYPRE);
	if(!vcrt_stereo_ensure()){
		vc_scene_end(VC_SC_EYPRE);
		return;
	}
	unsigned int fbo = vc_stereo_eye_fbo(eye);
	if(fbo == 0){
		vc_scene_end(VC_SC_EYPRE);
		return;
	}
	{
		// All shader families (librw + custom pipes) exist by the first world pass.
		static bool mvReported = false;
		if(!mvReported){ mvReported = true; vcrt_mv_pairs_report(); }
	}
	int w = 0, h = 0;
	vc_screen_size(&w, &h);
	vc_scene_end(VC_SC_EYPRE);

	vc_scene_begin(VC_SC_EYBIND);
	bindFramebuffer(fbo);
	glViewport(0, 0, w, h);
	vc_scene_end(VC_SC_EYBIND);
	vc_scene_begin(VC_SC_EYCLR);
	// Clear hygiene (multiview-plan.md 5.0b/4, risk L1): ANGLE turns this glClear
	// into the pass's loadAction=Clear ONLY if scissor is off, colour/depth/stencil
	// write masks are full and no draw happened yet. Anything else becomes a draw
	// (colour) or is skipped (depth/stencil under a zero mask) and the memoryless
	// attachment starts with Load or garbage -- measured on device 2026-09-25
	// (vc-mv5 inherited glDepthMask(FALSE): depth clear skipped). All three bits,
	// because D24S8 is one attachment: a colour+depth-only clear leaves stencil
	// on Load. Verified via [angle-vrr] passLoads/s (Clear/Clear/Clear).
	glDepthMask(GL_TRUE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDisable(GL_SCISSOR_TEST);
	glStencilMask(0xFFFFFFFFu);
	oldGlState.stencilWriteMask = 0xFFFFFFFFu;   // keep the low-level cache truthful
	// Clear the slice to the SKY colour (set per frame from CTimeCycle) instead of
	// black, so the sky fills the whole eye FOV as a world-anchored solid -- the
	// screen-space gradient/horizon band (which head-locked) is skipped in stereo.
	glClearColor(g_vcSkyClear[0], g_vcSkyClear[1], g_vcSkyClear[2], 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	// restore the depth-write state librw expects, and keep its cache in sync
	uint32 zmask = rwStateCache.zwrite ? GL_TRUE : GL_FALSE;
	glDepthMask(zmask);
	oldGlState.depthMask = zmask;
	vc_scene_end(VC_SC_EYCLR);
	vc_scene_begin(VC_SC_EYMTX);

	// Switches (read once). VC_STEREO_REAL: use the real compositor matrices
	// (default 1) vs the synthetic +/-0.5 m fallback, for A/B on device.
	// VC_STEREO_WORLD_SCALE: metres -> game units for the eye offset (GTA/VC is
	// ~1 unit per metre; tune on device). VC_STEREO_KEEP_PROJ: keep the GAME
	// projection even with real views -- isolates "offset wrong" from "FOV wrong".
	static int   sInit = 0, sReal = 1, sKeepProj = 0, sLogCtr = 0;
	static float sScale = 1.0f;
	if(!sInit){
		sInit = 1;
		const char *r = getenv("VC_STEREO_REAL");        if(r) sReal = atoi(r);
		const char *k = getenv("VC_STEREO_KEEP_PROJ");   if(k) sKeepProj = atoi(k);
		const char *s = getenv("VC_STEREO_WORLD_SCALE"); if(s) sScale = (float)atof(s);
		if(vcPerfLog()) printf("[vc-eyes-gl] REAL=%d KEEP_PROJ=%d WORLD_SCALE=%.4f (CANVAS: projected screen)\n",
		       sReal, sKeepProj, sScale);
	}

	vc_stereo_eye_matrices_t em;
	bool haveReal = sReal && vc_get_stereo_eye_matrices(&em) && em.valid;

	// Per-eye VIEW (vcMainView + this eye's half-IPD offset, metres * scale) and
	// PROJECTION (compositor FOV, symmetric, game-unit standard-Z depth). Shared by
	// the two-pass path (one eye per call) and the 5.1 one-pass path (both eyes into
	// the ViewID-indexed arrays). See the comments in the two-pass branch for WHY the
	// projection is rebuilt (reverse-Z trap, symmetric slices).
	auto eyeMatrices = [&](int me, float *vo, float *po) {
		memcpy(vo, vcMainView, 16*sizeof(float));
		if(haveReal){
			simd_float4 tL = em.view[0].columns[3];
			simd_float4 tR = em.view[1].columns[3];
			simd_float4 tM = 0.5f * (tL + tR);
			simd_float4 te = em.view[me].columns[3];
			vo[12] += (te.x - tM.x) * sScale;
			vo[13] += (te.y - tM.y) * sScale;
			vo[14] += (te.z - tM.z) * sScale;
			if(sKeepProj){
				memcpy(po, vcMainProj, 16*sizeof(float));
			}else{
				const float *cp = (const float *)&em.projection[me];  // column-major
				memset(po, 0, 16*sizeof(float));
				po[0]  = cp[0];         // FOV x
				po[5]  = cp[5];         // FOV y
				float n = vcMainNear, f = vcMainFar, invz = 1.0f / (f - n);
				po[10] = (f + n) * invz;         // standard-Z (near->-1, far->+1)
				po[11] = 1.0f;
				po[14] = -2.0f * n * f * invz;
				po[15] = 0.0f;
			}
		}else{
			// Fallback (VC_STEREO_REAL=0, or valid=0 in cinema/loading): synthetic
			// +/-0.5 m offset + game projection. Kept for A/B comparison.
			vo[12] += (me == 0) ? -0.5f : 0.5f;
			memcpy(po, vcMainProj, 16*sizeof(float));
		}
	};

	float v[16], p[16];
	if(vc_multiview_active()){
		// 5.1 ONE PASS: both eyes' matrices into the ViewID-indexed arrays of the
		// multiview twins (header.vert: u_viewMV[gl_ViewID_OVR]); the mono uniforms
		// and the getters carry the CENTRE view + the (symmetric) slice projection for
		// the CPU readers (game camera, cull, head-forward, CalcScreenCoors).
		float va[32], pa[32];
		eyeMatrices(0, &va[0],  &pa[0]);
		eyeMatrices(1, &va[16], &pa[16]);
		setUniform(u_viewMV, va);
		setUniform(u_projMV, pa);
		memcpy(g_vcEyeViewMV, va, sizeof(va));
		g_vcEyeViewMVValid = 1;
		memcpy(v, vcMainView, sizeof(v));
		memcpy(p, &pa[0], sizeof(p));
		setViewMatrix(v);
		setProjectionMatrix(p);
		{
			static int sDiagFrame = 0;
			if(++sDiagFrame % 90 == 0)
				printf("[vc-eye-mv] UPLOAD arrays: p0=%.4f p5=%.4f  view.x-off eye0=%.4f eye1=%.4f (centre view in mono uniform)\n",
				       pa[0], pa[5], va[12] - v[12], va[28] - v[12]);
		}
	}else{
		eyeMatrices(eye, v, p);
		setViewMatrix(v);
		// Per-eye upload proof, throttled, eyes 0/1 back-to-back. Row 0 is symmetric by
		// design: p0 = +compositor FOV x, p8 = p12 = 0 (the compositor adds the off-axis
		// itself). view.x-off = the IPD offset actually applied (opposite per eye).
		if(haveReal){
			static int sDiagFrame = 0;
			if(eye == 0) sDiagFrame++;
			if(sDiagFrame % 90 == 0){
				simd_float4 te = em.view[eye].columns[3];
				simd_float4 tM = 0.5f*(em.view[0].columns[3] + em.view[1].columns[3]);
				const float *cp = (const float *)&em.projection[eye];
				printf("[vc-eye%d] UPLOAD row0=[p0=%.4f p8=%.4f p12=%.4f]  view.x-off=%.4f  raw cp[p0=%.4f p8=%.4f]\n",
				       eye, p[0], p[8], p[12], te.x - tM.x, cp[0], cp[8]);
			}
		}
		setProjectionMatrix(p);
	}

	(void)sLogCtr;

	// Stash the VIEW/PROJECTION this pass uploaded (two-pass: this eye; one-pass:
	// centre view + slice projection), so the reVC side can set TheCamera to the same
	// camera (CPU-side sky/coronas/lighting read TheCamera instead of the GPU uniform).
	memcpy(g_vcEyeView, v, sizeof(v));
	memcpy(g_vcEyeProj, p, sizeof(p));
	g_vcEyeViewValid = 1;
	vc_scene_end(VC_SC_EYMTX);
}

// One-pass diagnostics: the PER-EYE views the multiview twins render with (librw
// convention), i.e. what a CPU-projected sprite WOULD need per eye. 0 unless the
// one-pass render is active.
extern "C" int
vc_get_eye_view_mv(int eye, float m[16])
{
	if(!g_vcEyeViewMVValid || !vc_multiview_active() || (eye != 0 && eye != 1)) return 0;
	memcpy(m, &g_vcEyeViewMV[16*eye], 16*sizeof(float));
	return 1;
}

// The eye VIEW (librw convention, column-major) that the last vc_stereo_eye_pass
// uploaded. Returns 1 + fills m if valid this session.
extern "C" int
vc_get_eye_view(float m[16])
{
	if(!g_vcEyeViewValid) return 0;
	// 5.0b/5 getter semantics: in the one-pass render there is no "last eye" --
	// CPU readers (cull head pose, head-forward, CalcScreenCoors, game camera)
	// get the CENTRE view (head x game, before the IPD offset). The projection
	// getter below stays: the slice projection is symmetric, equal for both eyes.
	if(vc_multiview_active()){
		memcpy(m, vcMainView, 16*sizeof(float));
		return 1;
	}
	memcpy(m, g_vcEyeView, 16*sizeof(float));
	return 1;
}

// The eye PROJECTION (librw clip, column-major) the last eye pass uploaded -- what
// the GPU world used. Lets the reVC side check CalcScreenCoors' fixed x/z*W mapping
// against the real slice projection.
extern "C" int
vc_get_eye_proj(float m[16])
{
	if(!g_vcEyeViewValid) return 0;
	memcpy(m, g_vcEyeProj, 16*sizeof(float));
	return 1;
}

// Rebind the cinema back-buffer FBO + viewport and restore the mono main
// matrices, so the rest of the frame (2D/HUD/menu, publish) continues into the
// cinema buffer. In stereo the cinema buffer is REPURPOSED as the head-locked
// HUD layer: the world already went into the eye slices, so we CLEAR the cinema
// buffer to fully TRANSPARENT (0,0,0,0) here -- wiping the sky drawn earlier by
// DoRWStuffStartOfFrame_Horizon -- so only the 2D/HUD/menu drawn afterwards
// remains, over transparency. The eye passes used their own depth; the cinema
// depth is cleared too so the 2D pass starts clean.
extern "C" void
vc_stereo_restore_main(void)
{
	// VC_MSAA: the last eye still sits in the shared multisample FBO -- resolve it
	// into its slice before we rebind the cinema/HUD buffer. No-op when MSAA is off.
	vc_stereo_msaa_resolve_pending();
	unsigned int fbo = vc_external_framebuffer();
	int w = 0, h = 0;
	vc_screen_size(&w, &h);
	bindFramebuffer(fbo);
	glViewport(0, 0, w, h);
	// Force ALPHA writes on for the clear AND the following 2D pass. Measured: the
	// clear leaves RGB=0 but alpha=255 (the "transparent" clear did not clear alpha
	// -> HUD buffer opaque -> premultiplied HUD quad blacks out the world). librw
	// never touches glColorMask, so the alpha channel must be write-masked by the
	// ANGLE/EGL default; force it explicitly. Scissor off too, so the clear covers
	// the whole buffer. Kept on afterwards so the 2D/HUD pass writes coverage alpha.
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDisable(GL_SCISSOR_TEST);
	glDepthMask(GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);   // transparent HUD background
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	uint32 zmask = rwStateCache.zwrite ? GL_TRUE : GL_FALSE;
	glDepthMask(zmask);
	oldGlState.depthMask = zmask;
	setViewMatrix(vcMainView);
	setProjectionMatrix(vcMainProj);
}

#endif

static void
endUpdate(Camera *cam)
{
}

static void
clearCamera(Camera *cam, RGBA *col, uint32 mode)
{
	RGBAf colf;
	GLbitfield mask;

	setFrameBuffer(cam);

	// make sure we're only clearing the part of the framebuffer
	// that is subrastered
	bool setScissor = cam->frameBuffer != cam->frameBuffer->parent;
	if(setScissor){
		Rect r = getFramebufferRect(cam->frameBuffer);
		glScissor(r.x, r.y, r.w, r.h);
		glEnable(GL_SCISSOR_TEST);
	}

	convColor(&colf, col);
	glClearColor(colf.red, colf.green, colf.blue, colf.alpha);
	mask = 0;
	if(mode & Camera::CLEARIMAGE)
		mask |= GL_COLOR_BUFFER_BIT;
	if(mode & Camera::CLEARZ)
		mask |= GL_DEPTH_BUFFER_BIT;
	if(mode & Camera::CLEARSTENCIL)
		mask |= GL_STENCIL_BUFFER_BIT;
	glDepthMask(GL_TRUE);
	glClear(mask);
	glDepthMask(rwStateCache.zwrite);

	if(setScissor)
		glDisable(GL_SCISSOR_TEST);
}

static void
showRaster(Raster *raster, uint32 flags)
{
//	glViewport(raster->offsetX, raster->offsetY,
//		raster->width, raster->height);

#if defined(LIBRW_VISIONOS)
	// TODO(visionos): no SwapBuffers; the GL result is blitted to a Metal texture outside librw.
	(void)flags;
#elif defined(LIBRW_SDL2)
	if(flags & Raster::FLIPWAITVSYNCH)
		SDL_GL_SetSwapInterval(1);
	else
		SDL_GL_SetSwapInterval(0);
	SDL_GL_SwapWindow(glGlobals.window);
#else
	if(flags & Raster::FLIPWAITVSYNCH)
		glfwSwapInterval(1);
	else
		glfwSwapInterval(0);
	glfwSwapBuffers(glGlobals.window);
#endif
}

static bool32
rasterRenderFast(Raster *raster, int32 x, int32 y)
{
	Raster *src = raster;
	Raster *dst = Raster::getCurrentContext();
	Gl3Raster *natdst = PLUGINOFFSET(Gl3Raster, dst, nativeRasterOffset);
	Gl3Raster *natsrc = PLUGINOFFSET(Gl3Raster, src, nativeRasterOffset);

	switch(dst->type){
	case Raster::NORMAL:
	case Raster::TEXTURE:
	case Raster::CAMERATEXTURE:
		switch(src->type){
		case Raster::CAMERA:
			setActiveTexture(0);
			glBindTexture(GL_TEXTURE_2D, natdst->texid);
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, x, (dst->height-src->height)-y,
				0, 0, src->width, src->height);
			glBindTexture(GL_TEXTURE_2D, boundTexture[0]);
			return 1;
		}
		break;
	}
	return 0;
}

#if defined(LIBRW_VISIONOS)

static void
makeVideoModeList(void)
{
	// TODO(visionos): expose exactly one mode; there is no monitor/mode enumeration.
	// Size comes from the skel over the C seam (single source; getFramebufferRect
	// reports this as the CAMERA framebuffer size).
	int vcW = 0, vcH = 0;
	vc_screen_size(&vcW, &vcH);
	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, 1, ID_DRIVER | MEMDUR_EVENT);
	glGlobals.modes[0].mode.width = vcW;
	glGlobals.modes[0].mode.height = vcH;
	glGlobals.modes[0].depth = 32;
	glGlobals.modes[0].flags = 0;
	glGlobals.numModes = 1;
}

static int
openVisionOS(EngineOpenParams *openparams)
{
	// No window/display init; the ANGLE GL context is created and made current
	// outside librw. EngineOpenParams.window carries ANGLE's eglGetProcAddress
	// (a void* here) which startVisionOS() uses to load the GL entry points.
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.window = openparams->window; // = eglGetProcAddress (see startVisionOS)

	memset(&gl3Caps, 0, sizeof(gl3Caps));
	// TODO(visionos): ANGLE exposes GLES; assume GLES 3.0 for now.
	gl3Caps.gles = 1;
	gl3Caps.glversion = 30;

	makeVideoModeList();

	return 1;
}

static int
closeVisionOS(void)
{
	// TODO(visionos): nothing to tear down; the context's lifetime is owned externally.
	return 1;
}

static int
startVisionOS(void)
{
	// The GLES context is created and made current by the host (skel/visionos)
	// via ANGLE; librw does not create it. It only needs the GL entry points,
	// which we load through the host-provided getProcAddress that arrived in
	// EngineOpenParams.window and is now cached in glGlobals.window.
	GLADloadproc getProcAddress = (GLADloadproc)glGlobals.window;
	if (getProcAddress == nil) {
		// printf in addition to RWERROR: Engine::start ignores DEVICEINIT's
		// return, so without this the failure would be swallowed.
		printf("[vc-gl] FAIL startVisionOS: no getProcAddress via EngineOpenParams.window\n");
		RWERROR((ERR_GENERAL, "visionOS: no getProcAddress supplied via EngineOpenParams.window"));
		return 0;
	}

	if (!gladLoadGLES2Loader(getProcAddress, gl3Caps.glversion)) {
		printf("[vc-gl] FAIL startVisionOS: gladLoadGLES2Loader failed\n");
		RWERROR((ERR_GENERAL, "visionOS: gladLoadGLES2Loader failed"));
		return 0;
	}

	printf("[vc-gl] OpenGL version: %s\n", glGetString(GL_VERSION));

	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopVisionOS(void)
{
	// TODO(visionos): context teardown is external; nothing to do.
	return 1;
}

#elif defined(LIBRW_SDL2)

static void
addVideoMode(int displayIndex, int modeIndex)
{
	int i;
	SDL_DisplayMode mode;

	SDL_GetDisplayMode(displayIndex, modeIndex, &mode);

	for(i = 1; i < glGlobals.numModes; i++){
		if(glGlobals.modes[i].mode.w == mode.w &&
		   glGlobals.modes[i].mode.h == mode.h &&
		   glGlobals.modes[i].mode.format == mode.format){
			// had this mode already, remember highest refresh rate
			if(mode.refresh_rate > glGlobals.modes[i].mode.refresh_rate)
				glGlobals.modes[i].mode.refresh_rate = mode.refresh_rate;
			return;
		}
	}

	// none found, add
	glGlobals.modes[glGlobals.numModes].mode = mode;
	glGlobals.modes[glGlobals.numModes].flags = VIDEOMODEEXCLUSIVE;
	glGlobals.numModes++;
}

static void
makeVideoModeList(int displayIndex)
{
	int i, num, depth;

	num = SDL_GetNumDisplayModes(displayIndex);
	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, num+1, ID_DRIVER | MEMDUR_EVENT);

	SDL_GetCurrentDisplayMode(displayIndex, &glGlobals.modes[0].mode);
	glGlobals.modes[0].flags = 0;
	glGlobals.numModes = 1;

	for(i = 0; i < num; i++)
		addVideoMode(displayIndex, i);

	for(i = 0; i < glGlobals.numModes; i++){
		depth = SDL_BITSPERPIXEL(glGlobals.modes[i].mode.format);
		// set depth to power of two
		for(glGlobals.modes[i].depth = 1; glGlobals.modes[i].depth < depth; glGlobals.modes[i].depth <<= 1);
	}
}

static int
openSDL2(EngineOpenParams *openparams)
{
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.pWindow = openparams->window;

	memset(&gl3Caps, 0, sizeof(gl3Caps));

	/* Init SDL */
	if(SDL_InitSubSystem(SDL_INIT_VIDEO)){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}

	makeVideoModeList(0);

	return 1;
}

static int
closeSDL2(void)
{
	SDL_QuitSubSystem(SDL_INIT_VIDEO);
	return 1;
}

static struct {
	int gl;
	int major, minor;
} profiles[] = {
	{ SDL_GL_CONTEXT_PROFILE_CORE, 3, 3 },
	{ SDL_GL_CONTEXT_PROFILE_CORE, 2, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 3, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 2, 0 },
	{ 0, 0, 0 },
};

static int
startSDL2(void)
{
	SDL_Window *win;
	SDL_GLContext ctx;
	DisplayMode *mode;

	mode = &glGlobals.modes[glGlobals.currentMode];

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, glGlobals.numSamples);

	int i;
	for(i = 0; profiles[i].gl; i++){
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, profiles[i].gl);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, profiles[i].major);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, profiles[i].minor);

		if(mode->flags & VIDEOMODEEXCLUSIVE) {
			win = SDL_CreateWindow(glGlobals.winTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, mode->mode.w, mode->mode.h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN);
			if (win)
				SDL_SetWindowDisplayMode(win, &mode->mode);
		} else {
			win = SDL_CreateWindow(glGlobals.winTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, glGlobals.winWidth, glGlobals.winHeight, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
			if (win)
				SDL_SetWindowDisplayMode(win, NULL);
		}
		if(win){
			gl3Caps.gles = profiles[i].gl == SDL_GL_CONTEXT_PROFILE_ES;
			gl3Caps.glversion = profiles[i].major*10 + profiles[i].minor;
			break;
		}
	}
	if(win == nil){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}
	ctx = SDL_GL_CreateContext(win);

	if (!((gl3Caps.gles ? gladLoadGLES2Loader : gladLoadGLLoader) ((GLADloadproc) SDL_GL_GetProcAddress, gl3Caps.glversion)) ) {
		RWERROR((ERR_GENERAL, "gladLoadGLLoader failed"));
		SDL_GL_DeleteContext(ctx);
		SDL_DestroyWindow(win);
		return 0;
	}

	printf("OpenGL version: %s\n", glGetString(GL_VERSION));

	glGlobals.window = win;
	glGlobals.glcontext = ctx;
	*glGlobals.pWindow = win;
	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopSDL2(void)
{
	SDL_GL_DeleteContext(glGlobals.glcontext);
	SDL_DestroyWindow(glGlobals.window);
	return 1;
}
#else

static void
addVideoMode(const GLFWvidmode *mode)
{
	int i;

	for(i = 1; i < glGlobals.numModes; i++){
		if(glGlobals.modes[i].mode.width == mode->width &&
		   glGlobals.modes[i].mode.height == mode->height &&
		   glGlobals.modes[i].mode.redBits == mode->redBits &&
		   glGlobals.modes[i].mode.greenBits == mode->greenBits &&
		   glGlobals.modes[i].mode.blueBits == mode->blueBits){
			// had this mode already, remember highest refresh rate
			if(mode->refreshRate > glGlobals.modes[i].mode.refreshRate)
				glGlobals.modes[i].mode.refreshRate = mode->refreshRate;
			return;
		}
	}

	// none found, add
	glGlobals.modes[glGlobals.numModes].mode = *mode;
	glGlobals.modes[glGlobals.numModes].flags = VIDEOMODEEXCLUSIVE;
	glGlobals.numModes++;
}

static void
makeVideoModeList(void)
{
	int i, num;
	const GLFWvidmode *modes;

	modes = glfwGetVideoModes(glGlobals.monitor, &num);
	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, num+1, ID_DRIVER | MEMDUR_EVENT);

	glGlobals.modes[0].mode = *glfwGetVideoMode(glGlobals.monitor);
	glGlobals.modes[0].flags = 0;
	glGlobals.numModes = 1;

	for(i = 0; i < num; i++)
		addVideoMode(&modes[i]);

	for(i = 0; i < glGlobals.numModes; i++){
		num = glGlobals.modes[i].mode.redBits +
			glGlobals.modes[i].mode.greenBits +
			glGlobals.modes[i].mode.blueBits;
		// set depth to power of two
		for(glGlobals.modes[i].depth = 1; glGlobals.modes[i].depth < num; glGlobals.modes[i].depth <<= 1);
	}
}

static int
openGLFW(EngineOpenParams *openparams)
{
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.pWindow = openparams->window;

	memset(&gl3Caps, 0, sizeof(gl3Caps));

	/* Init GLFW */
	if(!glfwInit()){
		RWERROR((ERR_GENERAL, "glfwInit() failed"));
		return 0;
	}

	glGlobals.monitor = glfwGetMonitors(&glGlobals.numMonitors)[0];

	makeVideoModeList();

	return 1;
}

static int
closeGLFW(void)
{
	glfwTerminate();
	return 1;
}

static void
glfwerr(int error, const char *desc)
{
	fprintf(stderr, "GLFW Error: %s\n", desc);
}

static struct {
	int gl;
	int major, minor;
} profiles[] = {
	{ GLFW_OPENGL_API, 3, 3 },
	{ GLFW_OPENGL_API, 2, 1 },
	{ GLFW_OPENGL_ES_API, 3, 1 },
	{ GLFW_OPENGL_ES_API, 2, 0 },
	{ 0, 0, 0 },
};

static int
startGLFW(void)
{
	GLFWwindow *win;
	DisplayMode *mode;

	mode = &glGlobals.modes[glGlobals.currentMode];

	glfwSetErrorCallback(glfwerr);
	glfwWindowHint(GLFW_RED_BITS, mode->mode.redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->mode.greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->mode.blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->mode.refreshRate);
	
	// GLX will round up to 2x or 4x if you ask for multisampling on with 1 sample
	// So only apply the SAMPLES hint if we actually want multisampling
	if (glGlobals.numSamples > 1)
		glfwWindowHint(GLFW_SAMPLES, glGlobals.numSamples);

	int i;
	for(i = 0; profiles[i].gl; i++){
		glfwWindowHint(GLFW_CLIENT_API, profiles[i].gl);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, profiles[i].major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, profiles[i].minor);

		if(mode->flags & VIDEOMODEEXCLUSIVE)
			win = glfwCreateWindow(mode->mode.width, mode->mode.height, glGlobals.winTitle, glGlobals.monitor, nil);
		else
			win = glfwCreateWindow(glGlobals.winWidth, glGlobals.winHeight, glGlobals.winTitle, nil, nil);
		if(win){
			gl3Caps.gles = profiles[i].gl == GLFW_OPENGL_ES_API;
			gl3Caps.glversion = profiles[i].major*10 + profiles[i].minor;
			break;
		}
	}
	if(win == nil){
		RWERROR((ERR_GENERAL, "glfwCreateWindow() failed"));
		return 0;
	}
	glfwMakeContextCurrent(win);

	/* Init GLAD */
	if (!((gl3Caps.gles ? gladLoadGLES2Loader : gladLoadGLLoader) ((GLADloadproc) glfwGetProcAddress, gl3Caps.glversion)) ) {
		RWERROR((ERR_GENERAL, "gladLoadGLLoader failed"));
		glfwDestroyWindow(win);
		return 0;
	}

	printf("OpenGL version: %s\n", glGetString(GL_VERSION));

	glGlobals.window = win;
	*glGlobals.pWindow = win;
	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopGLFW(void)
{
	glfwDestroyWindow(glGlobals.window);
	return 1;
}
#endif

static int
initOpenGL(void)
{
/*
	// this only works from 3.0 onward,
	// but luckily GLAD has already taken care of extensions for us
	int numExt;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);
	for(int i = 0; i < numExt; i++){
		const char *ext = (const char*)glGetStringi(GL_EXTENSIONS, i);
		if(ext == nil)
			continue;	// apparently that can happen...
		if(strcmp(ext, "GL_EXT_texture_compression_s3tc") == 0)
			gl3Caps.dxtSupported = true;
		else if(strcmp(ext, "GL_KHR_texture_compression_astc_ldr") == 0)
			gl3Caps.astcSupported = true;
//		printf("%d %s\n", i, ext);
	}
*/
	gl3Caps.dxtSupported = !!GLAD_GL_EXT_texture_compression_s3tc;
	gl3Caps.astcSupported = !!GLAD_GL_KHR_texture_compression_astc_ldr;

	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &gl3Caps.maxAnisotropy);
#ifdef LIBRW_VISIONOS
	// Measurement (NPC-face bleed / shimmer): does ANGLE-on-Metal honour anisotropic
	// filtering? cap==1 -> candidate "raise anisotropy" is dead; cap>=16 -> cheapest lever.
	if(getenv("VC_TEXLOG")) printf("[vc-tex] maxAnisotropy cap = %.1f\n", gl3Caps.maxAnisotropy);
#endif

	if(gl3Caps.gles){
		// TODO(visionos): librw kannte nur 100es und 310es; ANGLE liefert GLES 3.0
		if(gl3Caps.glversion >= 31)
			shaderDecl = shaderDecl310es;
		else if(gl3Caps.glversion >= 30)
			shaderDecl = shaderDecl300es;
		else
			shaderDecl = shaderDecl100es;
	}else{
		if(gl3Caps.glversion >= 30)
			shaderDecl = shaderDecl330;
		else
			shaderDecl = shaderDecl120;
	}

#ifndef RW_GL_USE_UBOS
	u_alphaRef = registerUniform("u_alphaRef", UNIFORM_VEC4);
	u_fogData = registerUniform("u_fogData", UNIFORM_VEC4);
	u_fogColor = registerUniform("u_fogColor", UNIFORM_VEC4);
#ifdef LIBRW_VISIONOS
	u_fogMode = registerUniform("u_fogMode", UNIFORM_VEC4);
	{
		// VC_FOG_RADIAL=1: fog by camera distance instead of view depth. Planar fog under
		// a ~110 deg per-eye FOV made far buildings vanish when looked at straight on and
		// reappear at the edge of view (device, rain, 2026-09-28). Default OFF until the
		// device run confirms; then flip.
		const char *e = getenv("VC_FOG_RADIAL");
		float fm[4] = { (e && e[0] == '0') ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f };   // default RADIAL
		setUniform(u_fogMode, fm);
		printf("[vc-fog] mode = %s (VC_FOG_RADIAL=%s)\n", fm[0] > 0.5f ? "RADIAL (camera distance)" : "PLANAR (view depth, stock)", e ? e : "unset");
	}
#endif
	u_proj = registerUniform("u_proj", UNIFORM_MAT4);
	u_view = registerUniform("u_view", UNIFORM_MAT4);
#ifdef LIBRW_VISIONOS
	// Per-view matrices of the multiview twins (header.vert under VC_MULTIVIEW).
	// Location -1 in mono programs -> flushUniforms skips them; fed from 5.1 on.
	u_projMV = registerUniform("u_projMV", UNIFORM_MAT4, 2);
	u_viewMV = registerUniform("u_viewMV", UNIFORM_MAT4, 2);
#endif
	u_world = registerUniform("u_world", UNIFORM_MAT4);
	u_ambLight = registerUniform("u_ambLight", UNIFORM_VEC4);
	u_lightParams = registerUniform("u_lightParams", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightPosition = registerUniform("u_lightPosition", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightDirection = registerUniform("u_lightDirection", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightColor = registerUniform("u_lightColor", UNIFORM_VEC4, MAX_LIGHTS);
	lastShaderUploaded = nil;
#else
	registerBlock("Scene");
	registerBlock("Object");
	registerBlock("State");
#endif
	u_matColor = registerUniform("u_matColor", UNIFORM_VEC4);
	u_surfProps = registerUniform("u_surfProps", UNIFORM_VEC4);

	// for im2d
	registerUniform("u_xform", UNIFORM_VEC4);

	glClearColor(0.25, 0.25, 0.25, 1.0);

	byte whitepixel[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glGenTextures(1, &whitetex);
	glBindTexture(GL_TEXTURE_2D, whitetex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1,
	             0, GL_RGBA, GL_UNSIGNED_BYTE, &whitepixel);

	resetRenderState();

	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);

	if(gl3Caps.glversion >= 30){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}

#ifdef RW_GL_USE_UBOS
	glGenBuffers(1, &ubo_state);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_state);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("State"), ubo_state);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), &uniformState,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &ubo_scene);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_scene);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("Scene"), ubo_scene);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), &uniformScene,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &ubo_object);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_object);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("Object"), ubo_object);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), &uniformObject,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
#endif

#include "shaders/default_vs_gl.inc"
#include "shaders/simple_fs_gl.inc"
	const char *vs[] = { shaderDecl, header_vert_src, default_vert_src, nil };
	const char *vs_fullLight[] = { shaderDecl, "#define DIRECTIONALS\n#define POINTLIGHTS\n#define SPOTLIGHTS\n", header_vert_src, default_vert_src, nil };
	const char *fs[] = { shaderDecl, header_frag_src, simple_frag_src, nil };
	const char *fs_noAT[] = { shaderDecl, "#define NO_ALPHATEST\n", header_frag_src, simple_frag_src, nil };

	defaultShader = Shader::create(vs, fs, "default");
	assert(defaultShader);
	defaultShader_noAT = Shader::create(vs, fs_noAT, "default_noAT");
	assert(defaultShader_noAT);

	defaultShader_fullLight = Shader::create(vs_fullLight, fs, "default_fullLight");
	assert(defaultShader_fullLight);
	defaultShader_fullLight_noAT = Shader::create(vs_fullLight, fs_noAT, "default_fullLight_noAT");
	assert(defaultShader_fullLight_noAT);

	openIm2D();
	openIm3D();

	return 1;
}

static int
termOpenGL(void)
{
	closeIm3D();
	closeIm2D();

	defaultShader->destroy();
	defaultShader = nil;
	defaultShader_noAT->destroy();
	defaultShader_noAT = nil;
	defaultShader_fullLight->destroy();
	defaultShader_fullLight = nil;
	defaultShader_fullLight_noAT->destroy();
	defaultShader_fullLight_noAT = nil;

	glDeleteTextures(1, &whitetex);
	whitetex = 0;

	return 1;
}

static int
finalizeOpenGL(void)
{
	return 1;
}

#if defined(LIBRW_VISIONOS)
static int
deviceSystemVisionOS(DeviceReq req, void *arg, int32 n)
{
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openVisionOS((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeVisionOS();

	case DEVICEINIT:
		return startVisionOS() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopVisionOS();

	case DEVICEFINALIZE:
		return finalizeOpenGL();

	// TODO(visionos): exactly one subsystem; there is no monitor enumeration.
	case DEVICEGETNUMSUBSYSTEMS:
		return 1;

	case DEVICEGETCURRENTSUBSYSTEM:
		return 0;

	case DEVICESETSUBSYSTEM:
		if(n >= 1)
			return 0;
		return 1;

	case DEVICEGETSUBSSYSTEMINFO:
		if(n >= 1)
			return 0;
		strncpy(((SubSystemInfo*)arg)->name, "visionOS", sizeof(SubSystemInfo::name));
		return 1;


	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.width;
		rwmode->height = glGlobals.modes[n].mode.height;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	// TODO(visionos): no multisample query yet; report a single level.
	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		return 1;
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#elif defined(LIBRW_SDL2)
static int
deviceSystemSDL2(DeviceReq req, void *arg, int32 n)
{
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openSDL2((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeSDL2();

	case DEVICEINIT:
		return startSDL2() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopSDL2();

	case DEVICEFINALIZE:
		return finalizeOpenGL();

	// TODO: implement subsystems

	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.w;
		rwmode->height = glGlobals.modes[n].mode.h;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		{
			GLint maxSamples;
			glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
			if(maxSamples == 0)
				return 1;
			return maxSamples;
		}
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#else

static int
deviceSystemGLFW(DeviceReq req, void *arg, int32 n)
{
	GLFWmonitor **monitors;
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openGLFW((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeGLFW();

	case DEVICEINIT:
		return startGLFW() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopGLFW();

	case DEVICEFINALIZE:
		return finalizeOpenGL();


	case DEVICEGETNUMSUBSYSTEMS:
		return glGlobals.numMonitors;

	case DEVICEGETCURRENTSUBSYSTEM:
		return glGlobals.currentMonitor;

	case DEVICESETSUBSYSTEM:
		monitors = glfwGetMonitors(&glGlobals.numMonitors);
		if(n >= glGlobals.numMonitors)
			return 0;
		glGlobals.currentMonitor = n;
		glGlobals.monitor = monitors[glGlobals.currentMonitor];
		return 1;

	case DEVICEGETSUBSSYSTEMINFO:
		monitors = glfwGetMonitors(&glGlobals.numMonitors);
		if(n >= glGlobals.numMonitors)
			return 0;
		strncpy(((SubSystemInfo*)arg)->name, glfwGetMonitorName(monitors[n]), sizeof(SubSystemInfo::name));
		return 1;


	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.width;
		rwmode->height = glGlobals.modes[n].mode.height;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		{
			GLint maxSamples;
			glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
			if(maxSamples == 0)
				return 1;
			return maxSamples;
		}
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#endif

Device renderdevice = {
	-1.0f, 1.0f,
	gl3::beginUpdate,
	gl3::endUpdate,
	gl3::clearCamera,
	gl3::showRaster,
	gl3::rasterRenderFast,
	gl3::setRenderState,
	gl3::getRenderState,
	gl3::im2DRenderLine,
	gl3::im2DRenderTriangle,
	gl3::im2DRenderPrimitive,
	gl3::im2DRenderIndexedPrimitive,
	gl3::im3DTransform,
	gl3::im3DRenderPrimitive,
	gl3::im3DRenderIndexedPrimitive,
	gl3::im3DEnd,
#if defined(LIBRW_VISIONOS)
	gl3::deviceSystemVisionOS
#elif defined(LIBRW_SDL2)
	gl3::deviceSystemSDL2
#else
	gl3::deviceSystemGLFW
#endif
};

}
}

#else
// urgh, probably should get rid of that eventually
#include "rwgl3.h"
namespace rw {
namespace gl3 { 
Gl3Caps gl3Caps;
bool32 needToReadBackTextures;
}
}
#endif
