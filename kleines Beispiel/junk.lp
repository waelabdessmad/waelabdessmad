\* Source Pyomo model name=unknown *\

min 
objective:
+12 sigma_ch(0_0)
+16 sigma_ch(0_1)
+12 sigma_ch(1_0)
+16 sigma_ch(1_1)
+12 sigma_ch(2_0)
+16 sigma_ch(2_1)
+12 sigma_ch(3_0)
+16 sigma_ch(3_1)
+12 sigma_ch(4_0)
+16 sigma_ch(4_1)
+12 sigma_ch(5_0)
+16 sigma_ch(5_1)
+12 sigma_ch(6_0)
+16 sigma_ch(6_1)
+12 sigma_ch(7_0)
+16 sigma_ch(7_1)
+12 sigma_ch(8_0)
+16 sigma_ch(8_1)
+12 sigma_ch(9_0)
+16 sigma_ch(9_1)
+8 sigma_dis(0_0)
+4 sigma_dis(0_1)
+8 sigma_dis(1_0)
+4 sigma_dis(1_1)
+8 sigma_dis(2_0)
+4 sigma_dis(2_1)
+8 sigma_dis(3_0)
+4 sigma_dis(3_1)
+8 sigma_dis(4_0)
+4 sigma_dis(4_1)
+8 sigma_dis(5_0)
+4 sigma_dis(5_1)
+8 sigma_dis(6_0)
+4 sigma_dis(6_1)
+8 sigma_dis(7_0)
+4 sigma_dis(7_1)
+8 sigma_dis(8_0)
+4 sigma_dis(8_1)
+8 sigma_dis(9_0)
+4 sigma_dis(9_1)

s.t.

c_l_dso_constraint(2)_:
+1 sigma_ch(2_0)
+1 sigma_ch(2_1)
-1 sigma_dis(2_0)
-1 sigma_dis(2_1)
>= 2

c_l_dso_constraint(3)_:
+1 sigma_ch(3_0)
+1 sigma_ch(3_1)
-1 sigma_dis(3_0)
-1 sigma_dis(3_1)
>= 1

c_l_dso_constraint(4)_:
+1 sigma_ch(4_0)
+1 sigma_ch(4_1)
-1 sigma_dis(4_0)
-1 sigma_dis(4_1)
>= 1

c_l_dso_constraint(6)_:
-1 sigma_ch(6_0)
-1 sigma_ch(6_1)
+1 sigma_dis(6_0)
+1 sigma_dis(6_1)
>= 1

c_l_dso_constraint(7)_:
-1 sigma_ch(7_0)
-1 sigma_ch(7_1)
+1 sigma_dis(7_0)
+1 sigma_dis(7_1)
>= 1

c_e_soc_evolution(1_0)_:
-0.94999999999999996 sigma_ch(1_0)
+1.0526315789473684 sigma_dis(1_0)
-1 sigma_soc(0_0)
+1 sigma_soc(1_0)
= 0

c_e_soc_evolution(1_1)_:
-0.90000000000000002 sigma_ch(1_1)
+1.1111111111111112 sigma_dis(1_1)
-1 sigma_soc(0_1)
+1 sigma_soc(1_1)
= 0

c_e_soc_evolution(2_0)_:
-0.94999999999999996 sigma_ch(2_0)
+1.0526315789473684 sigma_dis(2_0)
-1 sigma_soc(1_0)
+1 sigma_soc(2_0)
= 0

c_e_soc_evolution(2_1)_:
-0.90000000000000002 sigma_ch(2_1)
+1.1111111111111112 sigma_dis(2_1)
-1 sigma_soc(1_1)
+1 sigma_soc(2_1)
= 0

c_e_soc_evolution(3_0)_:
-0.94999999999999996 sigma_ch(3_0)
+1.0526315789473684 sigma_dis(3_0)
-1 sigma_soc(2_0)
+1 sigma_soc(3_0)
= 0

c_e_soc_evolution(3_1)_:
-0.90000000000000002 sigma_ch(3_1)
+1.1111111111111112 sigma_dis(3_1)
-1 sigma_soc(2_1)
+1 sigma_soc(3_1)
= 0

c_e_soc_evolution(4_0)_:
-0.94999999999999996 sigma_ch(4_0)
+1.0526315789473684 sigma_dis(4_0)
-1 sigma_soc(3_0)
+1 sigma_soc(4_0)
= 0

c_e_soc_evolution(4_1)_:
-0.90000000000000002 sigma_ch(4_1)
+1.1111111111111112 sigma_dis(4_1)
-1 sigma_soc(3_1)
+1 sigma_soc(4_1)
= 0

c_e_soc_evolution(5_0)_:
-0.94999999999999996 sigma_ch(5_0)
+1.0526315789473684 sigma_dis(5_0)
-1 sigma_soc(4_0)
+1 sigma_soc(5_0)
= 0

c_e_soc_evolution(5_1)_:
-0.90000000000000002 sigma_ch(5_1)
+1.1111111111111112 sigma_dis(5_1)
-1 sigma_soc(4_1)
+1 sigma_soc(5_1)
= 0

c_e_soc_evolution(6_0)_:
-0.94999999999999996 sigma_ch(6_0)
+1.0526315789473684 sigma_dis(6_0)
-1 sigma_soc(5_0)
+1 sigma_soc(6_0)
= 0

c_e_soc_evolution(6_1)_:
-0.90000000000000002 sigma_ch(6_1)
+1.1111111111111112 sigma_dis(6_1)
-1 sigma_soc(5_1)
+1 sigma_soc(6_1)
= 0

c_e_soc_evolution(7_0)_:
-0.94999999999999996 sigma_ch(7_0)
+1.0526315789473684 sigma_dis(7_0)
-1 sigma_soc(6_0)
+1 sigma_soc(7_0)
= 0

c_e_soc_evolution(7_1)_:
-0.90000000000000002 sigma_ch(7_1)
+1.1111111111111112 sigma_dis(7_1)
-1 sigma_soc(6_1)
+1 sigma_soc(7_1)
= 0

c_e_soc_evolution(8_0)_:
-0.94999999999999996 sigma_ch(8_0)
+1.0526315789473684 sigma_dis(8_0)
-1 sigma_soc(7_0)
+1 sigma_soc(8_0)
= 0

c_e_soc_evolution(8_1)_:
-0.90000000000000002 sigma_ch(8_1)
+1.1111111111111112 sigma_dis(8_1)
-1 sigma_soc(7_1)
+1 sigma_soc(8_1)
= 0

c_e_soc_evolution(9_0)_:
-0.94999999999999996 sigma_ch(9_0)
+1.0526315789473684 sigma_dis(9_0)
-1 sigma_soc(8_0)
+1 sigma_soc(9_0)
= 0

c_e_soc_evolution(9_1)_:
-0.90000000000000002 sigma_ch(9_1)
+1.1111111111111112 sigma_dis(9_1)
-1 sigma_soc(8_1)
+1 sigma_soc(9_1)
= 0

c_e_soc_evolution_init(0)_:
+1 sigma_soc(0_0)
= 0

c_e_soc_evolution_init(1)_:
+1 sigma_soc(0_1)
= 0

c_u_bat_max_power_in(0_0)_:
+1 sigma_ch(0_0)
<= 3

c_u_bat_max_power_in(0_1)_:
+1 sigma_ch(0_1)
<= 4

c_u_bat_max_power_in(1_0)_:
+1 sigma_ch(1_0)
<= 3

c_u_bat_max_power_in(1_1)_:
+1 sigma_ch(1_1)
<= 4

c_u_bat_max_power_in(2_0)_:
+1 sigma_ch(2_0)
<= 3

c_u_bat_max_power_in(2_1)_:
+1 sigma_ch(2_1)
<= 4

c_u_bat_max_power_in(3_0)_:
+1 sigma_ch(3_0)
<= 3

c_u_bat_max_power_in(3_1)_:
+1 sigma_ch(3_1)
<= 4

c_u_bat_max_power_in(4_0)_:
+1 sigma_ch(4_0)
<= 3

c_u_bat_max_power_in(4_1)_:
+1 sigma_ch(4_1)
<= 4

c_u_bat_max_power_in(5_0)_:
+1 sigma_ch(5_0)
<= 3

c_u_bat_max_power_in(5_1)_:
+1 sigma_ch(5_1)
<= 4

c_u_bat_max_power_in(6_0)_:
+1 sigma_ch(6_0)
<= 3

c_u_bat_max_power_in(6_1)_:
+1 sigma_ch(6_1)
<= 4

c_u_bat_max_power_in(7_0)_:
+1 sigma_ch(7_0)
<= 3

c_u_bat_max_power_in(7_1)_:
+1 sigma_ch(7_1)
<= 4

c_u_bat_max_power_in(8_0)_:
+1 sigma_ch(8_0)
<= 3

c_u_bat_max_power_in(8_1)_:
+1 sigma_ch(8_1)
<= 4

c_u_bat_max_power_in(9_0)_:
+1 sigma_ch(9_0)
<= 3

c_u_bat_max_power_in(9_1)_:
+1 sigma_ch(9_1)
<= 4

c_u_bat_max_power_out(0_0)_:
+1 sigma_dis(0_0)
<= 3

c_u_bat_max_power_out(0_1)_:
+1 sigma_dis(0_1)
<= 4

c_u_bat_max_power_out(1_0)_:
+1 sigma_dis(1_0)
<= 3

c_u_bat_max_power_out(1_1)_:
+1 sigma_dis(1_1)
<= 4

c_u_bat_max_power_out(2_0)_:
+1 sigma_dis(2_0)
<= 3

c_u_bat_max_power_out(2_1)_:
+1 sigma_dis(2_1)
<= 4

c_u_bat_max_power_out(3_0)_:
+1 sigma_dis(3_0)
<= 3

c_u_bat_max_power_out(3_1)_:
+1 sigma_dis(3_1)
<= 4

c_u_bat_max_power_out(4_0)_:
+1 sigma_dis(4_0)
<= 3

c_u_bat_max_power_out(4_1)_:
+1 sigma_dis(4_1)
<= 4

c_u_bat_max_power_out(5_0)_:
+1 sigma_dis(5_0)
<= 3

c_u_bat_max_power_out(5_1)_:
+1 sigma_dis(5_1)
<= 4

c_u_bat_max_power_out(6_0)_:
+1 sigma_dis(6_0)
<= 3

c_u_bat_max_power_out(6_1)_:
+1 sigma_dis(6_1)
<= 4

c_u_bat_max_power_out(7_0)_:
+1 sigma_dis(7_0)
<= 3

c_u_bat_max_power_out(7_1)_:
+1 sigma_dis(7_1)
<= 4

c_u_bat_max_power_out(8_0)_:
+1 sigma_dis(8_0)
<= 3

c_u_bat_max_power_out(8_1)_:
+1 sigma_dis(8_1)
<= 4

c_u_bat_max_power_out(9_0)_:
+1 sigma_dis(9_0)
<= 3

c_u_bat_max_power_out(9_1)_:
+1 sigma_dis(9_1)
<= 4

c_u_bat_max_soc(0_0)_:
+1 sigma_soc(0_0)
<= 10

c_u_bat_max_soc(0_1)_:
+1 sigma_soc(0_1)
<= 20

c_u_bat_max_soc(1_0)_:
+1 sigma_soc(1_0)
<= 10

c_u_bat_max_soc(1_1)_:
+1 sigma_soc(1_1)
<= 20

c_u_bat_max_soc(2_0)_:
+1 sigma_soc(2_0)
<= 10

c_u_bat_max_soc(2_1)_:
+1 sigma_soc(2_1)
<= 20

c_u_bat_max_soc(3_0)_:
+1 sigma_soc(3_0)
<= 10

c_u_bat_max_soc(3_1)_:
+1 sigma_soc(3_1)
<= 20

c_u_bat_max_soc(4_0)_:
+1 sigma_soc(4_0)
<= 10

c_u_bat_max_soc(4_1)_:
+1 sigma_soc(4_1)
<= 20

c_u_bat_max_soc(5_0)_:
+1 sigma_soc(5_0)
<= 10

c_u_bat_max_soc(5_1)_:
+1 sigma_soc(5_1)
<= 20

c_u_bat_max_soc(6_0)_:
+1 sigma_soc(6_0)
<= 10

c_u_bat_max_soc(6_1)_:
+1 sigma_soc(6_1)
<= 20

c_u_bat_max_soc(7_0)_:
+1 sigma_soc(7_0)
<= 10

c_u_bat_max_soc(7_1)_:
+1 sigma_soc(7_1)
<= 20

c_u_bat_max_soc(8_0)_:
+1 sigma_soc(8_0)
<= 10

c_u_bat_max_soc(8_1)_:
+1 sigma_soc(8_1)
<= 20

c_u_bat_max_soc(9_0)_:
+1 sigma_soc(9_0)
<= 10

c_u_bat_max_soc(9_1)_:
+1 sigma_soc(9_1)
<= 20

c_u_bat_s_ch(0_0)_:
+1 sigma_ch(0_0)
+1.5000000000000004 sigma_soc(0_0)
<= 15.000000000000004

c_u_bat_s_ch(0_1)_:
+1 sigma_ch(0_1)
+1.0000000000000002 sigma_soc(0_1)
<= 20.000000000000004

c_u_bat_s_ch(1_0)_:
+1 sigma_ch(1_0)
+1.5000000000000004 sigma_soc(1_0)
<= 15.000000000000004

c_u_bat_s_ch(1_1)_:
+1 sigma_ch(1_1)
+1.0000000000000002 sigma_soc(1_1)
<= 20.000000000000004

c_u_bat_s_ch(2_0)_:
+1 sigma_ch(2_0)
+1.5000000000000004 sigma_soc(2_0)
<= 15.000000000000004

c_u_bat_s_ch(2_1)_:
+1 sigma_ch(2_1)
+1.0000000000000002 sigma_soc(2_1)
<= 20.000000000000004

c_u_bat_s_ch(3_0)_:
+1 sigma_ch(3_0)
+1.5000000000000004 sigma_soc(3_0)
<= 15.000000000000004

c_u_bat_s_ch(3_1)_:
+1 sigma_ch(3_1)
+1.0000000000000002 sigma_soc(3_1)
<= 20.000000000000004

c_u_bat_s_ch(4_0)_:
+1 sigma_ch(4_0)
+1.5000000000000004 sigma_soc(4_0)
<= 15.000000000000004

c_u_bat_s_ch(4_1)_:
+1 sigma_ch(4_1)
+1.0000000000000002 sigma_soc(4_1)
<= 20.000000000000004

c_u_bat_s_ch(5_0)_:
+1 sigma_ch(5_0)
+1.5000000000000004 sigma_soc(5_0)
<= 15.000000000000004

c_u_bat_s_ch(5_1)_:
+1 sigma_ch(5_1)
+1.0000000000000002 sigma_soc(5_1)
<= 20.000000000000004

c_u_bat_s_ch(6_0)_:
+1 sigma_ch(6_0)
+1.5000000000000004 sigma_soc(6_0)
<= 15.000000000000004

c_u_bat_s_ch(6_1)_:
+1 sigma_ch(6_1)
+1.0000000000000002 sigma_soc(6_1)
<= 20.000000000000004

c_u_bat_s_ch(7_0)_:
+1 sigma_ch(7_0)
+1.5000000000000004 sigma_soc(7_0)
<= 15.000000000000004

c_u_bat_s_ch(7_1)_:
+1 sigma_ch(7_1)
+1.0000000000000002 sigma_soc(7_1)
<= 20.000000000000004

c_u_bat_s_ch(8_0)_:
+1 sigma_ch(8_0)
+1.5000000000000004 sigma_soc(8_0)
<= 15.000000000000004

c_u_bat_s_ch(8_1)_:
+1 sigma_ch(8_1)
+1.0000000000000002 sigma_soc(8_1)
<= 20.000000000000004

c_u_bat_s_ch(9_0)_:
+1 sigma_ch(9_0)
+1.5000000000000004 sigma_soc(9_0)
<= 15.000000000000004

c_u_bat_s_ch(9_1)_:
+1 sigma_ch(9_1)
+1.0000000000000002 sigma_soc(9_1)
<= 20.000000000000004

c_u_bat_s_dis(0_0)_:
+1 sigma_dis(0_0)
-3 sigma_soc(0_0)
<= 0

c_u_bat_s_dis(0_1)_:
+1 sigma_dis(0_1)
-2 sigma_soc(0_1)
<= 0

c_u_bat_s_dis(1_0)_:
+1 sigma_dis(1_0)
-3 sigma_soc(1_0)
<= 0

c_u_bat_s_dis(1_1)_:
+1 sigma_dis(1_1)
-2 sigma_soc(1_1)
<= 0

c_u_bat_s_dis(2_0)_:
+1 sigma_dis(2_0)
-3 sigma_soc(2_0)
<= 0

c_u_bat_s_dis(2_1)_:
+1 sigma_dis(2_1)
-2 sigma_soc(2_1)
<= 0

c_u_bat_s_dis(3_0)_:
+1 sigma_dis(3_0)
-3 sigma_soc(3_0)
<= 0

c_u_bat_s_dis(3_1)_:
+1 sigma_dis(3_1)
-2 sigma_soc(3_1)
<= 0

c_u_bat_s_dis(4_0)_:
+1 sigma_dis(4_0)
-3 sigma_soc(4_0)
<= 0

c_u_bat_s_dis(4_1)_:
+1 sigma_dis(4_1)
-2 sigma_soc(4_1)
<= 0

c_u_bat_s_dis(5_0)_:
+1 sigma_dis(5_0)
-3 sigma_soc(5_0)
<= 0

c_u_bat_s_dis(5_1)_:
+1 sigma_dis(5_1)
-2 sigma_soc(5_1)
<= 0

c_u_bat_s_dis(6_0)_:
+1 sigma_dis(6_0)
-3 sigma_soc(6_0)
<= 0

c_u_bat_s_dis(6_1)_:
+1 sigma_dis(6_1)
-2 sigma_soc(6_1)
<= 0

c_u_bat_s_dis(7_0)_:
+1 sigma_dis(7_0)
-3 sigma_soc(7_0)
<= 0

c_u_bat_s_dis(7_1)_:
+1 sigma_dis(7_1)
-2 sigma_soc(7_1)
<= 0

c_u_bat_s_dis(8_0)_:
+1 sigma_dis(8_0)
-3 sigma_soc(8_0)
<= 0

c_u_bat_s_dis(8_1)_:
+1 sigma_dis(8_1)
-2 sigma_soc(8_1)
<= 0

c_u_bat_s_dis(9_0)_:
+1 sigma_dis(9_0)
-3 sigma_soc(9_0)
<= 0

c_u_bat_s_dis(9_1)_:
+1 sigma_dis(9_1)
-2 sigma_soc(9_1)
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= sigma_ch(0_0) <= +inf
   0 <= sigma_ch(0_1) <= +inf
   0 <= sigma_ch(1_0) <= +inf
   0 <= sigma_ch(1_1) <= +inf
   0 <= sigma_ch(2_0) <= +inf
   0 <= sigma_ch(2_1) <= +inf
   0 <= sigma_ch(3_0) <= +inf
   0 <= sigma_ch(3_1) <= +inf
   0 <= sigma_ch(4_0) <= +inf
   0 <= sigma_ch(4_1) <= +inf
   0 <= sigma_ch(5_0) <= +inf
   0 <= sigma_ch(5_1) <= +inf
   0 <= sigma_ch(6_0) <= +inf
   0 <= sigma_ch(6_1) <= +inf
   0 <= sigma_ch(7_0) <= +inf
   0 <= sigma_ch(7_1) <= +inf
   0 <= sigma_ch(8_0) <= +inf
   0 <= sigma_ch(8_1) <= +inf
   0 <= sigma_ch(9_0) <= +inf
   0 <= sigma_ch(9_1) <= +inf
   0 <= sigma_dis(0_0) <= +inf
   0 <= sigma_dis(0_1) <= +inf
   0 <= sigma_dis(1_0) <= +inf
   0 <= sigma_dis(1_1) <= +inf
   0 <= sigma_dis(2_0) <= +inf
   0 <= sigma_dis(2_1) <= +inf
   0 <= sigma_dis(3_0) <= +inf
   0 <= sigma_dis(3_1) <= +inf
   0 <= sigma_dis(4_0) <= +inf
   0 <= sigma_dis(4_1) <= +inf
   0 <= sigma_dis(5_0) <= +inf
   0 <= sigma_dis(5_1) <= +inf
   0 <= sigma_dis(6_0) <= +inf
   0 <= sigma_dis(6_1) <= +inf
   0 <= sigma_dis(7_0) <= +inf
   0 <= sigma_dis(7_1) <= +inf
   0 <= sigma_dis(8_0) <= +inf
   0 <= sigma_dis(8_1) <= +inf
   0 <= sigma_dis(9_0) <= +inf
   0 <= sigma_dis(9_1) <= +inf
   0 <= sigma_soc(0_0) <= +inf
   0 <= sigma_soc(0_1) <= +inf
   0 <= sigma_soc(1_0) <= +inf
   0 <= sigma_soc(1_1) <= +inf
   0 <= sigma_soc(2_0) <= +inf
   0 <= sigma_soc(2_1) <= +inf
   0 <= sigma_soc(3_0) <= +inf
   0 <= sigma_soc(3_1) <= +inf
   0 <= sigma_soc(4_0) <= +inf
   0 <= sigma_soc(4_1) <= +inf
   0 <= sigma_soc(5_0) <= +inf
   0 <= sigma_soc(5_1) <= +inf
   0 <= sigma_soc(6_0) <= +inf
   0 <= sigma_soc(6_1) <= +inf
   0 <= sigma_soc(7_0) <= +inf
   0 <= sigma_soc(7_1) <= +inf
   0 <= sigma_soc(8_0) <= +inf
   0 <= sigma_soc(8_1) <= +inf
   0 <= sigma_soc(9_0) <= +inf
   0 <= sigma_soc(9_1) <= +inf
end
