dtmc
module RelAIBotiX_NoDone
  s : [0..9] init 0;

  [] s=0 -> 1.0176e-08 : (s'=5) + 0.999999989824 : (s'=1);
  [] s=1 -> 4.779e-09 : (s'=6) + 0.730769227277 : (s'=2) + 0.269230767944 : (s'=0);
  [] s=2 -> 4.523e-09 : (s'=7) + 0.999999995477 : (s'=3);
  [] s=3 -> 3.553e-09 : (s'=8) + 0.999999996447 : (s'=4);
  [] s=4 -> 9.979e-09 : (s'=9) + 0.999999990021 : (s'=0);
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
  s=0 : 1.3759259259285999;
  s=1 : 1.0200000000028955;
  s=2 : 0.6894736842132977;
  s=3 : 0.8505263157926115;
  s=4 : 1.238421052636123;
endrewards
