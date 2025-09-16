dtmc
module RelAIBotiX_NoDone
  s : [0..1] init 0;

  [] s=0 -> 1.0017e-08 : (s'=1) + 0.999999989983 : (s'=0);
  [] s=1 -> 1 : (s'=1);
endmodule

label "Init_state" = s=0;
label "Init_failure" = s=1;
label "failure" = s=1;

rewards "time"
  s=0 : 9.977500000030412;
endrewards
