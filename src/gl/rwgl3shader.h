#ifdef RW_OPENGL

namespace rw {
namespace gl3 {

// TODO: make this dynamic
enum {
	MAX_UNIFORMS = 40,
	MAX_BLOCKS = 20
};

enum UniformType
{
	UNIFORM_NA,	// managed by the user
	UNIFORM_VEC4,
	UNIFORM_IVEC4,
	UNIFORM_MAT4
};

struct Uniform
{
	char *name;
	UniformType type;
	//bool dirty;
	uint32 serialNum;
	int32 num;
	void *data;
};

struct UniformRegistry
{
	int32 numUniforms;
	Uniform uniforms[MAX_UNIFORMS];

	int32 numBlocks;
	char *blockNames[MAX_BLOCKS];
};

int32 registerUniform(const char *name, UniformType type = UNIFORM_NA, int32 num = 1);
int32 findUniform(const char *name);
int32 registerBlock(const char *name);
int32 findBlock(const char *name);

void setUniform(int32 id, void *data);
void flushUniforms(void);

extern UniformRegistry uniformRegistry;

struct Shader
{
	GLuint program;
	// same number of elements as UniformRegistry::numUniforms
	GLint *uniformLocations;
	uint32 *serialNums;
	int32 numUniforms;	// just to be sure!
	// visionOS (multiview-plan.md 5.0b/1): the multiview twin of this program --
	// same sources + OVR_multiview2 prelude + VC_MULTIVIEW. nil unless created
	// with a pair name under VC_MULTIVIEW=1. Not selected by use() yet (5.0b/2).
	Shader *mv;

	// mvName != nil marks a shader that can run in the world pass and therefore
	// needs a multiview twin (ANGLE checks the view count exactly: one program
	// cannot serve mono and 2-view FBOs). Post-effects pass nil.
	static Shader *create(const char **vsrc, const char **fsrc, const char *mvName = nil);
//	static Shader *fromFiles(const char *vs, const char *fs);
//	static Shader *fromStrings(const char *vsrc, const char *fsrc);
	void use(void);
	void destroy(void);
};

extern Shader *currentShader;
#ifdef LIBRW_VISIONOS
// 5.0b/2: set by bindFramebuffer() when the bound FBO is the registered multiview
// FBO (vc_multiview_fbo(), 0 until Stufe 5.1). Shader::use() then selects the twin.
extern bool vcMultiviewBound;
extern unsigned g_vcUseMono, g_vcUseMv;   // use() selections per frame (report)
#endif

}
}

#endif
