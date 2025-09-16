dtmc
module RelAIBotiX_NoDone
  s : [0..9] init 0;

  [] s=0 -> 7.96e-09 : (s'=5) + 0.99999999204 : (s'=1);
  [] s=1 -> 4.445e-09 : (s'=6) + 0.863636359798 : (s'=2) + 0.136363635757 : (s'=0);
  [] s=2 -> 5.096e-09 : (s'=7) + 0.999999994904 : (s'=3);
  [] s=3 -> 3.084e-09 : (s'=8) + 0.999999996916 : (s'=4);
  [] s=4 -> 9.368e-09 : (s'=9) + 0.999999990632 : (s'=0);
  [] s=5 -> 1 : (s'=5);
  [] s=6 -> 1 : (s'=6);
  [] s=7 -> 1 : (s'=7);
  [] s=8 -> 1 : (s'=8);
  [] s=9 -> 1 : (s'=9);
endmodule

label "Init_state" = s=0;
label "Move" = s=1;
label "Pick" = s=2;
label "Carry" = s=3;
label "Place" = s=4;
label "Init_failure" = s=5;
label "Move_failure" = s=6;
label "Pick_failure" = s=7;
label "Carry_failure" = s=8;
label "Place_failure" = s=9;
label "failure" = s=5 | s=6 | s=7 | s=8 | s=9;

rewards "steps"
	true : 1;
endrewards

rewards "time"
  s=0 : 1.1556521739150047;
  s=1 : 1.0659090909111866;
  s=2 : 0.7963157894754179;
  s=3 : 0.7510526315804531;
  s=4 : 1.2257894736866906;
endrewards
