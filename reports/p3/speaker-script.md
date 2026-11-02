# AstraGO P3：约 7 分钟中英对照讲稿

对应最新版 12 页 PPT。英文约 780 词，建议按每分钟约 115 词讲述，并留出自然停顿。以下各页时间是排练参考，合计 7 分钟。中文用于理解和排练；若两种语言都完整读出，总时长将超过 7 分钟。讲稿已同步写入 PPT 备注区。

第 2 页中的旧版平面尺寸、轮距和离地间隙仍标为待核对；确认后需同步调整该页和讲稿。

---

## 第 1 页｜封面与汇报主线（约 30 秒）

**English**

Good morning. Today we present the detailed design of AstraGO, our autonomous water transport vehicle. It uses a single-level frame made from forty-by-forty millimetre aluminium profiles and two independently driven rear wheels. This is our current chassis CAD model, before arm and electronics integration. I will explain the mechanical design, our Orin Nano debugging progress, and the remaining integration work.

**中文对照**

大家好，今天我们汇报 AstraGO 自动运水车的详细设计。我们采用单层 4040 铝型材车架，两个后轮分别由独立电机驱动。图中是目前的底盘 CAD 模型，尚未加入机械臂和电子设备。接下来，我会介绍机械结构、Orin Nano 的调试进展，以及后续需要完成的集成工作。

---

## 第 2 页｜车架尺寸（约 35 秒）

**English**

The frame is now single-level, using forty-by-forty profiles. The earlier drawing gives an eight-hundred-by-three-hundred-and-fifty millimetre frame plan, a seven-hundred-and-fifty millimetre track, and one hundred millimetres of ground clearance. These dimensions still need checking against the updated CAD. We have not transferred the old two-level heights or tyre diameter. The frame footprint is also different from the complete vehicle envelope, which includes the wheels, suspension, arm and bowl.

**中文对照**

现在车架已经确定为单层 4040 型材。旧图中的车架平面尺寸是 800×350 毫米，轮距是 750 毫米，离地间隙是 100 毫米，这些数值还需要与更新后的 CAD 核对。原来双层结构的高度和轮径没有直接沿用。另外，车架平面尺寸不等于整车外廓，还需要考虑轮胎、悬架、机械臂和水碗伸出的范围。

---

## 第 3 页｜车架连接与受力（约 35 秒）

**English**

This detail shows the frame connections. The longitudinal rails support the equipment, while crossmembers connect the rails and provide mounting locations. Brackets and braced supports connect the wheel modules. Our proposed arm adapter should distribute its load across both rails through the crossmembers. Next, we need to define hole patterns, plate thickness and fasteners. The profile size alone does not establish the strength of the assembled frame.

**中文对照**

这张局部图展示车架的连接方式。纵梁承担设备载荷，横梁连接两侧纵梁，并提供安装位置。轮端模块则通过支架和斜撑接入车架。机械臂转接件计划通过横梁，把载荷分散到两侧纵梁。下一步需要确定孔位、板厚和紧固件，不能仅凭型材尺寸，就认定整个车架的强度已经满足要求。

---

## 第 4 页｜悬架与转向（约 35 秒）

**English**

The front assembly must accommodate suspension travel and steering together. We need to check bump, rebound and steering extremes for interference between tyres, linkage, brackets and cables. The steering servo and final linkage geometry remain open. At the rear, the carriers must retain wheel direction while allowing suspension movement. The CAD shows our current arrangement; checking the combined motion envelope is the next step.

**中文对照**

前轮总成需要同时满足悬架运动和转向。我们要检查悬架压缩、回弹以及极限转向角度下，轮胎、连杆、支架和线缆之间是否干涉。转向舵机和最终连杆几何尚未确定。后轮轮座则需要在允许悬架运动的同时约束车轮朝向。这里展示的是当前布局，下一步要检查这些运动叠加后的空间范围。

---

## 第 5 页｜后轮电机总成（约 35 秒）

**English**

Each rear wheel uses this compact drive module. The supplier identifies a sixty-three-seventy-four brushless outrunner motor, with one thousand watts of listed power, thirty-six to forty-eight volts, and an eight-inch pneumatic wheel. These are supplier specifications. The advertised forty-five kilometres per hour is not a measured speed for our vehicle. We still need the exact motor configuration and compatible controller details before integration.

**中文对照**

两个后轮分别采用这种紧凑的驱动模块。商家标注的电机是 6374 无刷外转子，功率为 1000 瓦，电压为 36 到 48 伏，配有 8 寸充气轮。这些数值来自商家资料，其中宣传的最高时速 45 公里，并不是我们整车的实测速度。集成前还需要确认电机的具体配置，以及兼容的控制器参数。

---

## 第 6 页｜皮带传动（约 35 秒）

**English**

Power passes from the motor pulley, through the belt, to the wheel pulley. Each rear wheel has an independent drive path. Wheel speed equals motor speed divided by the transmission ratio, so pulley teeth counts and loaded motor speed must be confirmed. We also need to check belt alignment and tension through suspension travel. During turns, the control system must allow suitable differences between the left and right wheel speeds.

**中文对照**

动力从电机带轮经过皮带，传到车轮侧带轮，左右后轮各有独立的传动路径。轮速等于电机转速除以传动比，因此还需要确认带轮齿数和电机的带载转速。同时，要检查悬架运动过程中皮带的对齐和张紧情况。转弯时，左右后轮经过的路径长度不同，控制系统也需要允许两侧轮速合理变化。

---

## 第 7 页｜机械臂选型（约 35 秒）

**English**

We selected a Zhongling Technology open-source robotic arm. This is the supplier's original model, including its two-jaw gripper. Our team will design a four-finger gripper, but that CAD model is not complete yet. We still need to verify the arm's exact version, working reach, payload and control interface. Our integration work focuses on mounting the arm, developing the gripper and choosing a compact transport posture.

**中文对照**

我们选择了众灵科技的开源机械臂。这里是商家提供的原装模型，末端仍是原装两爪。我们准备自行设计四爪夹具，但四爪 CAD 目前还没有完成。后续需要核实机械臂的具体版本、工作范围、负载能力和控制接口。我们的集成重点是机械臂安装、夹具设计，以及适合运输的紧凑姿态。

---

## 第 8 页｜四爪夹具与取放流程（约 40 秒）

**English**

Our four-finger gripper will approach the bowl from above. The maximum bowl diameter is one hundred and fifty-six millimetres, and the bowl plus water weighs about zero point six five kilograms. Gripper and adapter mass must also count toward the arm's load. The sequence is locate, descend, grip, test lift, transport and place. Throughout transport, the arm must keep holding the bowl, without a vehicle tray supporting it.

**中文对照**

我们计划让四爪夹具从上方接近水碗。水碗最大直径为 156 毫米，碗和水合计约 0.65 千克。机械臂的负载还要加上夹具和转接件的重量。动作顺序是定位、下降、夹紧、小幅试提、运输和放置。整个运输过程都由机械臂持续夹持，不使用车载碗托承重。

---

## 第 9 页｜机械臂安装（约 35 秒）

**English**

We propose an adapter plate bridging both frame rails, designed around the arm base's actual hole pattern. Its location must provide access to the platform and bowl, while allowing a compact transport posture. The board, fasteners and cables also need service access. Mount position, plate thickness and bolt arrangement remain to be defined, considering the changing load and reach as the arm moves.

**中文对照**

我们计划根据机械臂底座的实际孔位，设计连接两侧纵梁的转接板。安装位置既要保证机械臂能够到达平台和水碗，也要允许它在运输时收拢。同时，还要保留控制板、紧固件和线缆的操作空间。最终位置、板厚和螺栓布置，需要结合机械臂运动时不断变化的载荷和工作范围来确定。

---

## 第 10 页｜Orin Nano 调试与接口（约 45 秒）

**English**

We have started bench debugging of the Jetson Orin Nano. This photograph shows our actual setup. Its planned role is perception and path planning; vehicle integration comes next. The proposed battery and protection system will supply the rear drive controllers, arm and regulated auxiliaries, including the Orin. Battery, controller, servo and camera choices remain open. Before connecting them, we must confirm power requirements, commands, feedback and fault handling. Debugging is underway; autonomous operation has not yet been verified.

**中文对照**

我们已经开始调试 Jetson Orin Nano，照片展示的就是实际调试现场。它计划负责感知和路径规划，下一步再接入整车。供电方面，拟由电池和保护系统向后轮驱动器、机械臂，以及包括 Orin 在内的辅助稳压电路供能。电池、控制器、舵机和相机仍未最终确定，连接前需要核实供电要求、指令、反馈和故障处理方式。目前正在进行调试，整车自主运行尚未得到验证。

---

## 第 11 页｜下一步验证（约 35 秒）

**English**

Our next steps are to confirm CAD dimensions and mounting clearances, check belt tension and wheel support, then test arm reach and continuous gripping with the intended load. After confirming the power interfaces, we will perform low-speed steering and controlled-stop tests. Evidence for the next review should include checked CAD, measured interfaces, a loaded-grip video and test logs. These are planned checks, not completed test results.

**中文对照**

下一步，先确认 CAD 尺寸和安装空间，再检查皮带张紧和车轮支撑，然后在预期载荷下测试机械臂的可达性与持续夹持能力。核对供电接口后，再开展低速转向和受控停车测试。下一次汇报应提供检查后的 CAD、接口实测尺寸、带载夹持视频和测试记录。这些是接下来要完成的验证，并不是已经取得的测试结果。

---

## 第 12 页｜总结（约 25 秒）

**English**

Our design now combines a single-level forty-by-forty frame, independent rear belt drives and a selected open-source arm. The four-finger gripper is still being modelled, and Orin Nano debugging has started. Our next milestone is checked mounts, reliable gripping and compatible interfaces. Thank you. We welcome your questions.

**中文对照**

目前，我们的方案采用单层 4040 车架、左右独立的后轮皮带驱动，以及选定的开源机械臂。四爪夹具还在建模，Orin Nano 已开始调试。下一阶段的目标是确认安装结构、验证可靠夹持，并打通各系统接口。谢谢，欢迎老师提问。
