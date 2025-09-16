dtmc
module RelAIBotiX
  s : [0..2] init 0;

  [] s=0 -> 1.0017e-08 : (s'=1) + 0.499999994991 : (s'=2) + 0.499999994992 : (s'=0);
  [] s=1 -> 1 : (s'=1);
  [] s=2 -> 1 : (s'=2);
endmodule

label "Init_failure" = s=1;
label "done" = s=2;
label "failure" = s=1;

rewards "steps"
  true : 1;
endrewards
