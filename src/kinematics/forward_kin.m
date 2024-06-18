function pos = forward_kin(in1,in2)
%FORWARD_KIN
%    POS = FORWARD_KIN(IN1,IN2)

%    This function was generated by the Symbolic Math Toolbox version 9.0.
%    17-Jun-2024 17:01:54

a1 = in1(:,1);
a2 = in1(:,2);
a3 = in1(:,3);
a4 = in1(:,4);
a5 = in1(:,5);
a6 = in1(:,6);
a7 = in1(:,7);
a8 = in1(:,8);
a9 = in1(:,9);
t1 = in2(:,1);
t2 = in2(:,2);
t3 = in2(:,3);
t4 = in2(:,4);
t5 = in2(:,5);
t6 = in2(:,6);
t7 = in2(:,7);
t8 = in2(:,8);
t10 = cos(t1);
t11 = cos(t2);
t12 = cos(t3);
t13 = cos(t4);
t14 = cos(t5);
t15 = cos(t6);
t16 = cos(t7);
t17 = cos(t8);
t18 = sin(t1);
t19 = sin(t2);
t20 = sin(t3);
t21 = sin(t4);
t22 = sin(t5);
t23 = sin(t6);
t24 = sin(t7);
t25 = sin(t8);
t26 = a1+a2;
t27 = a3+t26;
t28 = a1.*t18;
t29 = t10.*t11;
t30 = t10.*t19;
t31 = t11.*t18;
t32 = t18.*t19;
t33 = t10-1.0;
t34 = t11-1.0;
t35 = t12-1.0;
t36 = t13-1.0;
t37 = t14-1.0;
t38 = t15-1.0;
t39 = t16-1.0;
t40 = t17-1.0;
t41 = a4+t27;
t43 = -t32;
t45 = a1.*t33;
t47 = t26.*t30;
t48 = t26.*t32;
t52 = t10.*t26.*t34;
t53 = t30+t31;
t54 = t18.*t26.*t34;
t42 = a5+t41;
t50 = -t45;
t55 = -t52;
t56 = t29+t43;
t57 = t12.*t53;
t58 = t20.*t53;
t64 = t27.*t35.*t53;
t44 = a6+t42;
t59 = t12.*t56;
t60 = t20.*t56;
t61 = -t58;
t62 = t27.*t58;
t65 = t27.*t35.*t56;
t46 = a7+t44;
t63 = t27.*t60;
t66 = -t65;
t67 = t57+t60;
t68 = t59+t61;
t71 = -t13.*(t58-t59);
t72 = -t21.*(t58-t59);
t77 = -t36.*t41.*(t58-t59);
t78 = t36.*t41.*(t58-t59);
t49 = a8+t46;
t69 = t13.*t67;
t70 = t21.*t67;
t75 = t41.*t72;
t76 = t36.*t41.*t67;
t51 = a9+t49;
t73 = -t70;
t74 = t41.*t70;
t79 = t69+t72;
t83 = -t14.*(t70+t13.*(t58-t59));
t84 = -t22.*(t70+t13.*(t58-t59));
t89 = -t37.*t42.*(t70+t13.*(t58-t59));
t90 = t37.*t42.*(t70+t13.*(t58-t59));
t80 = t71+t73;
t81 = t14.*t79;
t82 = t22.*t79;
t87 = t42.*t84;
t88 = t37.*t42.*t79;
t85 = -t82;
t86 = t42.*t82;
t91 = t81+t84;
t95 = -t15.*(t82+t14.*(t70+t13.*(t58-t59)));
t96 = -t23.*(t82+t14.*(t70+t13.*(t58-t59)));
t101 = -t38.*t44.*(t82+t14.*(t70+t13.*(t58-t59)));
t102 = t38.*t44.*(t82+t14.*(t70+t13.*(t58-t59)));
t92 = t83+t85;
t93 = t15.*t91;
t94 = t23.*t91;
t99 = t44.*t96;
t100 = t38.*t44.*t91;
t97 = -t94;
t98 = t44.*t94;
t103 = t93+t96;
t107 = -t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59))));
t108 = -t24.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59))));
t113 = -t39.*t46.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59))));
t114 = t39.*t46.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59))));
t104 = t95+t97;
t105 = t16.*t103;
t106 = t24.*t103;
t111 = t46.*t108;
t112 = t39.*t46.*t103;
t109 = -t106;
t110 = t46.*t106;
t115 = t105+t108;
t116 = t107+t109;
mt1 = [0.0,a1,0.0,t28-t18.*t26,t50+t10.*t26,0.0,t28+t47+t54-t27.*t53,t48+t50+t55+t27.*t56,0.0,t28+t47+t54+t63+t64-t41.*t67,t48+t50+t55+t62+t66-t41.*(t58-t59),0.0,t28+t47+t54+t63+t64+t75+t76-t42.*t79,t48+t50+t55+t62+t66+t74+t78-t42.*(t70+t13.*(t58-t59)),0.0,t28+t47+t54+t63+t64+t75+t76+t87+t88-t44.*t91,t48+t50+t55+t62+t66+t74+t78+t86+t90-t44.*(t82+t14.*(t70+t13.*(t58-t59))),0.0,t28+t47+t54+t63+t64+t75+t76+t87+t88+t99+t100-t46.*t103,t48+t50+t55+t62+t66+t74+t78+t86+t90+t98+t102-t46.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59)))),0.0];
mt2 = [t28+t47+t54+t63+t64+t75+t76+t87+t88+t99+t100+t111+t112-t49.*t115,t48+t50+t55+t62+t66+t74+t78+t86+t90+t98+t102+t110+t114-t49.*(t106+t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59))))),0.0,t28+t47+t54+t63+t64+t75+t76+t87+t88+t99+t100+t111+t112+t51.*(t25.*(t106+t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59)))))-t17.*t115)-t25.*t49.*(t106+t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59)))))+t40.*t49.*t115];
mt3 = [t48+t50+t55+t62+t66+t74+t78+t86+t90+t98+t102+t110+t114-t51.*(t17.*(t106+t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59)))))+t25.*t115)+t40.*t49.*(t106+t16.*(t94+t15.*(t82+t14.*(t70+t13.*(t58-t59)))))+t25.*t49.*t115,0.0];
pos = reshape([mt1,mt2,mt3],3,9);
