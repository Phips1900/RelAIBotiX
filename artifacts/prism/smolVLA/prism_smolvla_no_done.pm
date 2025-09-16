dtmc
module RelAIBotiX_NoDone
  s : [0..9] init 0;

  [] s=0 -> 7.952e-09 : (s'=5) + 0.999999992048 : (s'=1);
  [] s=1 -> 6.048e-09 : (s'=6) + 0.599999996371 : (s'=2) + 0.399999997581 : (s'=0);
  [] s=2 -> 4.571e-09 : (s'=7) + 0.999999995429 : (s'=3);
  [] s=3 -> 2.964e-09 : (s'=8) + 0.999999997036 : (s'=4);
  [] s=4 -> 9.047e-09 : (s'=9) + 0.999999990953 : (s'=0);
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
  s=0 : 1.413333333337769;
  s=1 : 1.5276000000034549;
  s=2 : 0.7526666666691177;
  s=3 : 0.7626666666690293;
  s=4 : 1.1646666666703744;
endrewards
