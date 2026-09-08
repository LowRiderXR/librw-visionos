uniform sampler2D tex0;

FSIN vec4 v_color;
FSIN vec2 v_tex0;
FSIN float v_fog;

void
main(void)
{
#ifdef VC_SKIN_LODBIAS
	vec4 color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y), float(VC_SKIN_LODBIAS));
#else
	vec4 color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y));
#endif
	color.rgb = mix(u_fogColor.rgb, color.rgb, v_fog);
	DoAlphaTest(color.a);
	FRAGCOLOR(color);
}

