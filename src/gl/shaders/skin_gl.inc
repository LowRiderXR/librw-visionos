const char *skin_vert_src =
// VC_SKINFOG selects the fog depth component: z (original, desktop-correct) or w
// (view depth, matches default.vert). Default z keeps macOS byte-identical; the
// visionOS build injects "#define VC_SKINFOG w" (see gl3skin.cpp) because our
// per-eye rebuilt projection makes clip-Z the wrong fog input -> NPCs tinted to fog.
"#ifndef VC_SKINFOG\n"
"#define VC_SKINFOG z\n"
"#endif\n"
"uniform mat4 u_boneMatrices[64];\n"

"VSIN(ATTRIB_POS)	vec3 in_pos;\n"

"VSOUT vec4 v_color;\n"
"VSOUT vec2 v_tex0;\n"
"VSOUT float v_fog;\n"

"void\n"
"main(void)\n"
"{\n"
"	vec3 SkinVertex = vec3(0.0, 0.0, 0.0);\n"
"	vec3 SkinNormal = vec3(0.0, 0.0, 0.0);\n"
"	for(int i = 0; i < 4; i++){\n"
"		SkinVertex += (u_boneMatrices[int(in_indices[i])] * vec4(in_pos, 1.0)).xyz * in_weights[i];\n"
"		SkinNormal += (mat3(u_boneMatrices[int(in_indices[i])]) * in_normal) * in_weights[i];\n"
"	}\n"

"	vec4 Vertex = u_world * vec4(SkinVertex, 1.0);\n"
"	gl_Position = u_proj * u_view * Vertex;\n"
"	vec3 Normal = mat3(u_world) * SkinNormal;\n"

"	v_tex0 = in_tex0;\n"

"	v_color = in_color;\n"
"	v_color.rgb += u_ambLight.rgb*surfAmbient;\n"
"	v_color.rgb += DoDynamicLight(Vertex.xyz, Normal)*surfDiffuse;\n"
"	v_color = clamp(v_color, 0.0, 1.0);\n"
"	v_color *= u_matColor;\n"

"#ifdef VC_SKIN_NOFOG\n"
"	v_fog = 1.0;\n"
"#else\n"
"	v_fog = DoFog(gl_Position.VC_SKINFOG);\n"
"#endif\n"
"}\n"
;
