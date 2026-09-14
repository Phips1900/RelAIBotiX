# Reliability model and parameter provenance

This document freezes the assumptions used for the publication results in
`results/paper`. It describes a comparative, stress-adjusted reliability model;
the reported MTTF values are not experimentally validated field lifetimes.

## Additive normalized exposure

For component `j` during detected skill `k`, RelAIBotiX computes

```text
E[j,k] = sum(dt)
       + sum(abs(torque) / torque_reference * dt)
       + sum(abs(delta_position) / velocity_reference)
```

over the intervals in which that component is active. In mathematical form,

\[
E_{j,k}=\sum_{i\in\mathcal I_{j,k}}\Delta t_i
+\sum_{i\in\mathcal I_{j,k}}\frac{|\tau_{j,i}|}{\tau_{j,\mathrm{ref}}}\Delta t_i
+\sum_{i\in\mathcal I_{j,k}}\frac{|q_{j,i+1}-q_{j,i}|}{\dot q_{j,\mathrm{ref}}}.
\]

The references use 30% of the configured rated limits:

\[
\tau_{j,\mathrm{ref}}=0.3\tau_{j,\max},\qquad
\dot q_{j,\mathrm{ref}}=0.3\dot q_{j,\max}.
\]

All three terms have units of time. The third term is normalized integrated
absolute velocity, so velocity and traveled distance are not counted twice. The
model is additive: a stationary but operating component retains baseline exposure,
and short sensor peaks are not amplified by multiplying or cubing stress terms.
Missing telemetry omits only the unavailable term. Always-active electronics use
elapsed skill time.

The nominal component probability `p0` over basis duration `T0` is converted to a
constant base hazard rate and then applied to the effective exposure:

\[
\lambda_{j,0}=-\frac{\ln(1-p_{j,0})}{T_0},\qquad
p_{j,k}=1-\exp(-\lambda_{j,0}E_{j,k}).
\]

Component probabilities are combined by one fault tree per skill. The resulting
skill probabilities parameterize the empirical DTMC. Completed executions restart
in the repeated-operation model; modeled failures are absorbing. Component
sensitivity multiplies one nominal component probability by ten while holding all
other assumptions fixed.

## Nominal one-minute component probabilities

| Platform/component | Probability per minute | Derivation |
|---|---:|---|
| Franka joint, each of seven | 1.1904761199e-7 | Equal series allocation of the official 20,000 h arm lifetime |
| SO-ARM actuator, each joint and gripper | 3.3333327778e-7 | Generic 50,000 h servo bearing-life reference |
| Stretch wheel, lift, arm, or gripper actuator | 3.3333327778e-7 | Generic 50,000 h stepper/servo bearing-life reference |
| Stretch two-servo head assembly | 6.6666644444e-7 | Two single-actuator rates in series |
| Stretch three-servo wrist assembly | 9.9999950000e-7 | Three single-actuator rates in series |
| Franka gripper | 7.344e-8 | Prior FIDES/NPRD-based framework |
| Power supply | 6.674e-8 | Prior FIDES/NPRD-based framework |
| Camera, each nonredundant instance | 1.029e-9 | Legacy FIDES/NPRD-based parameterization |
| Controller, each copy | 2.167e-10 | Prior FIDES/NPRD-based framework; dual-controller structure is illustrative |

For the Franka joints,

\[
p_{\mathrm{joint},1\mathrm{min}}
=1-\exp\left(-\frac{1}{7\times20{,}000\times60}\right).
\]

For the generic 50,000 h actuator reference,

\[
p_{\mathrm{actuator},1\mathrm{min}}
=1-\exp\left(-\frac{1}{50{,}000\times60}\right).
\]

## Sources and limitations

1. Franka Emika, *Robot Product Manual*, expected lifetime and operating limits:
   <https://download.franka.de/Product-Manual-Franka-Emika-Robot_EN_10_2021.pdf>
2. Franka Robotics, control-parameter limits:
   <https://support.franka.de/docs/control_parameters.html>
3. Oriental Motor, generic servo and stepper bearing service-life estimates:
   <https://www.orientalmotor.com/support/service-life.html>
4. Hugging Face, official SO-101 assembly and six-actuator inventory:
   <https://huggingface.co/docs/lerobot/main/assemble_so101>
5. STS3215 specification, greater than 100,000 cycles at one-fifth stall torque.
   This is used only as a qualification plausibility check, not as an MTTF:
   <https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf>
6. Hello Robot, Stretch 3 hardware inventory:
   <https://docs-arch.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/>
7. FIDES reliability methodology:
   <https://www.fides-reliability.org/en/node/546>
8. P. Grimmeisen et al., “Automated and Continuous Risk Assessment for ROS-Based
   Software-Defined Robotic Systems,” CASE 2023:
   <https://doi.org/10.1109/CASE56687.2023.10260416>

Public field-failure tables for the evaluated robot joints were unavailable. The
actuator probabilities are therefore transparent engineering allocations and
cross-family nominal references, not manufacturer-certified joint failure rates.
The camera value still requires the exact legacy FIDES/NPRD worksheet to be
archived with the paper supplement. Until then, it should be described as inherited
from the preceding parameterization rather than independently re-derived.

Comparisons are strongest within the same platform under the same assumptions.
Cross-platform numerical comparisons must account for differing telemetry: current
SO-ARM and LIBERO recordings do not contain joint effort, whereas Franka and Stretch
recordings do.
