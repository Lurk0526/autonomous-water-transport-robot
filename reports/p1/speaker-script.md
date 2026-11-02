# P1 · 中英对照讲稿

对应文件：P1_AquaGo_V6.pptx（12页）。主讲：XU Xiangxi。

**台上只讲英文；中文和提示仅供理解。英文约797词，目标6分30秒。** 按约115–125词/分钟讲述并留出换页停顿，通常可落在6–7分钟；必须本人计时排练，不能以字数保证时长。每页时间是分配目标。

成员姓名和照片仍由你们填写。不要逐一念14名成员；每组介绍职责即可。项目尚处规划阶段，因此保留“plan / propose / candidate / will test”，不要改成已经实现。

## 01 · Introduction｜封面

目标23秒 · 累计0:23

**讲述提示（不读）：**先告诉导师：项目是什么、今天讲什么、重点是什么。

**English — 台上读**

Good morning. I am Xiangxi Xu. Our vehicle is called AquaGo. Today, I will explain the design and team roles for our autonomous water transport vehicle. Our focus is automation: picking up a full bowl, carrying it outdoors, and putting it down at the destination.

**中文理解**

大家早上好，我是XU Xiangxi。我们的小车叫AquaGo。今天我会介绍自动运水车的设计和团队分工。重点是自动化：夹起满水碗，完成户外运输，再把碗放到终点。

## 02 · Locomotion system｜行驶系

目标30秒 · 累计0:53

**讲述提示（不读）：**指底盘照片，再指35 km/h；强调载水直线，不是全程高速。

**English — 台上读**

We plan to buy a ZD9021 chassis without a motor. We will add a drive system with feedback. The vehicle must reach at least thirty-five kilometres per hour on a straight section while holding the full bowl. It can slow down for turns, obstacles, and rough ground. We will choose the motor for the vehicle with all its equipment and load.

**中文理解**

我们计划购买不带电机的ZD9021底盘，再增加带反馈的驱动系统。车辆必须在夹持满水碗的情况下，在直线段达到至少35 km/h。转弯、避障和路面颠簸时可以减速。电机将按安装全部设备并带载的整车选型。

## 03 · Motor and steering servo selection｜电机与转向舵机选型

目标60秒 · 累计1:53

**讲述提示（不读）：**电机：慢行可控、加减速平顺；舵机：少空动、回中准。

**English — 台上读**

For driving, we prefer a brushless motor with sensors and a suitable FOC driver. We read wheel speed so the vehicle can move slowly near a platform. Smooth torque and braking can help reduce sudden movement of the water. We will check speed under load, current, and heat. For steering, we need a digital positional servo with low backlash and good return-to-centre accuracy. This helps avoid repeated steering corrections. Its torque and power supply must match the steering load. We will also limit how quickly it turns. These are design needs; we still have to test them. Motor speed, gear ratio, and wheel size affect vehicle speed. More gear reduction makes the wheels turn slower but gives them more torque. We must test the full load.

**中文理解**

驱动方面，我们优先考虑带传感器的无刷电机和合适的FOC驱动器。读取轮速反馈，帮助车辆在平台附近缓慢行驶。平顺的转矩和制动有助于减少水的突然晃动。我们会检查带载速度、电流和发热。转向方面，需要回差小、回中精度好的数字位置舵机，以减少反复修正方向。它的转矩和供电要匹配转向负载，并限制转向变化速度。这些是设计要求，仍需测试。电机转速、传动比和轮胎大小共同影响车速。减速比更大时，轮子转得更慢，但轮端转矩更大；仍需带完整负载测试。

## 04 · Robotic arm｜机械臂

目标47秒 · 累计2:40

**讲述提示（不读）：**半成品机械臂外购，连接件与夹具自主设计；650g仅为碗和水。

**English — 台上读**

We will buy a semi-finished robotic arm. Our team will design the mounting and connecting parts, as well as the gripper. The bowl and water weigh about six hundred and fifty grams. The arm must also carry the gripper and the parts attached to its end. We will check joint torque, position feedback, and motor temperature while the arm holds the bowl. A small test lift will help check the grip before driving. The bowl has no lid and must stay in the arm throughout the trip. The arm model in this picture has not been selected.

**中文理解**

我们将购买半成品机械臂，由团队自主设计安装连接件以及夹具。碗和水约重650克，机械臂还要承担夹具和安装在末端的连接件重量。我们会检查关节转矩、位置反馈以及长时间夹持水碗时的电机温度。行驶前通过小幅试提辅助检查夹持。碗不加盖，运输全程由机械臂夹持。图片中的机械臂型号尚未选定。

## 05 · Vision and onboard computing｜视觉与上位计算

目标37秒 · 累计3:17

**讲述提示（不读）：**区分“找到平台”和“知道整条路怎么走”；提前建图是否允许，仍待确认。

**English — 台上读**

The D455 camera and Jetson Orin Nano eight-gigabyte computer are options we are considering. Vision will detect obstacles, boundaries, and platform markers. Markers help us find the platforms, but they do not show the whole route. We still need to ask whether we can build a map or record the route before the test. We will choose the navigation method after checking the rules. We also need outdoor tests of image quality, detection distance, and response time.

**中文理解**

D455相机与Jetson Orin Nano 8GB计算机是正在考虑的选项。视觉将识别障碍物、边界和平台标记。标记有助于找到平台，但不能给出完整路线。我们仍需确认是否允许在测试前建图或记录路线，再根据规则选择导航方式。另外还需要户外测试图像质量、识别距离和响应时间。

## 06 · STM32: the low-level control layer｜STM32底层控制

目标42秒 · 累计3:59

**讲述提示（不读）：**顺着三列讲：判断 → 指令与反馈 → 实际动作。

**English — 台上读**

We plan to use an STM32 for low-level control. The onboard computer handles vision and planning. The STM32 sends control commands regularly and reads motor and sensor data. If new commands stop arriving, it should trigger the planned stop response. It also checks driver faults. We plan to use a ready-made motor driver and the arm’s own controller. The STM32 sends control signals. It does not supply power directly to the motor. We will test what happens if communication is lost, as well as the hardware emergency stop.

**中文理解**

我们计划用STM32负责底层控制。上位计算机负责视觉和规划，STM32定期发送控制指令，读取电机与传感器数据。如果持续收不到新指令，应触发计划中的停车处理；它也会检查驱动器故障。计划使用现成的电机驱动器及机械臂自带控制器。STM32发送控制信号，不直接给电机供动力。我们会测试通信中断时的处理方式以及硬件急停。

## 07 · Automation connects the two systems｜自动化整合

目标49秒 · 累计4:48

**讲述提示（不读）：**讲完整取运放，再强调：看得够远、停得下来，才允许相应车速。

**English — 台上读**

The task starts by finding the first platform. The vehicle stops, the arm grips the bowl, and a small test lift helps check the grip. The vehicle then moves to the destination while the arm keeps holding the bowl. We will choose the speed based on clear space ahead, measured stopping distance, turns, and ground conditions. The vehicle stops again before putting the bowl down. If vision or the grip is not reliable, the task should pause. Remote takeover will be available. The control team will lead system integration. We will test the full task at low speed before raising the speed step by step.

**中文理解**

任务从找到起点平台开始。车辆停车，机械臂夹住碗，再通过小幅试提辅助检查夹持。随后车辆向终点行驶，机械臂始终夹着碗。速度将根据前方自由空间、实测停车距离、弯道和路面条件选择。放碗前车辆再次停车。如果视觉或夹持不可靠，任务应暂停，并允许遥控接管。控制组牵头系统集成。先低速测试完整任务，再逐步提速。

## 08 · Control Systems｜控制组

目标19秒 · 累计5:07

**讲述提示（不读）：**控制组牵头集成；讲清近期交付物，不逐个念成员姓名。

**English — 台上读**

The control team will lead system integration. It will connect the STM32, drive, steering, and arm controls. Its next tasks are to list the control interfaces and plan the bench tests. It will organise full-system tests.

**中文理解**

控制组牵头系统集成，连接STM32、驱动、转向和机械臂控制。近期任务是整理控制接口并规划台架测试，并组织整车测试。

## 09 · Vision & Perception｜视觉组

目标18秒 · 累计5:25

**讲述提示（不读）：**强调交付的是可用于控制的信息。

**English — 台上读**

The vision team will work on obstacle and boundary detection, target localisation, and camera calibration. Next, it will plan tests for platform markers and obstacles. The route-planning method still depends on the course rules.

**中文理解**

视觉组负责障碍与边界识别、目标定位和相机标定。下一步规划平台标记和障碍识别测试。路线规划方式仍需依据课程规则确定。

## 10 · Mechanical Design｜结构组

目标20秒 · 累计5:45

**讲述提示（不读）：**结构简化，仍要负责载荷、安装与稳定性。

**English — 台上读**

The mechanical team will design the mounting and connecting parts and the bowl gripper. It will fit the arm, camera, and electronics onto the chassis. Next, it will draw the layout, develop the gripper design, and estimate the loads.

**中文理解**

结构组自主设计安装连接件和碗夹具，将机械臂、相机与电子设备安装到底盘上。下一步绘制布置图、完善夹具设计并估算载荷。

## 11 · Finance & Procurement｜财务与采购组

目标24秒 · 累计6:09

**讲述提示（不读）：**汇报明确说课程预算上限10000元；不能把内部资金预留当成课程额度。

**English — 台上读**

The finance and procurement team will compare parts, prices, and delivery times. It will keep a parts list and all cost records. The course budget limit is ten thousand yuan. The cost plan must include transport, spare parts, and rework. Parts must meet the agreed technical needs.

**中文理解**

财务与采购组比较配件、价格和交期，维护配件表和全部成本记录。课程预算上限为10000元。成本计划需要包含运输、备件和返工，所购配件必须满足约定的技术要求。

## 12 · Closing｜结尾

目标21秒 · 累计6:30

**讲述提示（不读）：**停止展开新技术，留给导师提问。

**English — 台上读**

Our goal is to pick up, carry, and place the bowl automatically, while keeping as much water as possible. Next, we will check the route rules, choose suitable parts, and start bench tests. Thank you. I am happy to answer your questions.

**中文理解**

目标是自动夹起、运送和放下碗，并尽可能保留水量。下一步确认路线规则、选择合适配件并开始台架测试。谢谢，我愿意回答大家的问题。

## 排练与超时处理

1. 先看技术速读，再读稿，不要先背陌生缩写。
2. 第一遍边翻PPT边读英文，完整计时；不要把中文和提示读出来。
3. 目标6:20–6:40，给换页、停顿和紧张留余量。超过7分钟，优先删去第3页“We will check speed under load, current, and heat.”以及第3页“These are design needs; we still have to test them.”这两句；保留载水35 km/h、持续夹持、自动取运放和开发状态。
4. 不用快速念完所有技术指标。第3页讲“电机平顺、舵机精准”，第6页讲“谁判断、谁控制、谁供能”，其他细节由PPT和问答补充。
5. 若讲得不足6分钟，可以放慢、指向对应部件和流程，不临时编造参数或性能。

读法：STM32 = S-T-M thirty-two；FOC = F-O-C；PWM = P-W-M；D455 = D four-five-five；8GB = eight gigabytes；35 km/h = thirty-five kilometres per hour；650 g = six hundred and fifty grams。
