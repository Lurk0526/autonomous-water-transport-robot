# AstraGO · P2 中英文批注演讲稿

对应：AstraGO_P2_演讲版_含结束页.pptx。第1—11页主讲（含结束页），第12—13页备用。

**上台只读英文正文；中英文提示均不读。** 主讲共855词，按125词/分钟约6.8分钟，另留换页与停顿时间。请按课程实际时限排练。

## 01 · 开场与任务主线

建议约35秒。

### English speech · 英文演讲正文

Good morning. This is AstraGO, our autonomous water transport vehicle. The mission is to locate, grip, navigate and place. We identify the target, pick up the uncovered bowl from above, travel towards the destination marker while avoiding obstacles, and place the bowl at the end. Today we will explain the overall structure, four-finger gripper, control architecture and navigation strategy. The vehicle images are concept references, and our team drawings will replace them.

### 中文释义

大家好，这是我们的自动运水车AstraGO。任务主线是定位、夹取、导航和放置：识别目标，从上方向下夹起无盖水碗，朝目标标记物自主行驶并避障，最后将碗放下。本次介绍总体结构、四爪夹取、控制架构与导航策略。现有车架图是概念参考，之后替换为团队绘制的图。

### Delivery note · 讲述批注（不读）

**EN:** Point to the four mission words; pause between stages.

**中文：** 开场指向四个任务词，稍作停顿；不要逐一念底部参数。

---

## 02 · 总体布置

建议约35秒。

### English speech · 英文演讲正文

The package separates manipulation, computing and power. The arm mounting zone is near the front and spans both longitudinal rails. The camera needs a clear view of targets and obstacles. The computing unit sits on the upper level and the battery is low and central. Two rear motors provide drive, while the front wheels steer. Before fixing the mounting holes, we must check the arm, bowl, wheel and steering envelopes together.

### 中文释义

总体布置将操作、计算与供能分区。机械臂安装区域靠前，跨接两根纵梁；相机需要看清目标和障碍。计算单元布置在上层，电池布置在较低且靠中心的位置。后轮两台电机提供驱动，前轮转向。确定安装孔位前，需要综合检查机械臂、持碗、车轮和转向运动包络。这是布置方案，不是已经完成的装配验证。

### Delivery note · 讲述批注（不读）

**EN:** Point to each zone in order; present it as a proposed package.

**中文：** 按机械臂、相机、计算、电池、驱动顺序指图，强调布置仍可调整。

---

## 03 · 车架尺寸与高度定义

建议约44秒。

### English speech · 英文演讲正文

Both frame layers use twenty-five by twenty-five millimetre aluminium profiles, with an initial footprint of eight hundred by four hundred and fifty millimetres. The three heights refer to different endpoints. A is the complete frame height, from the lower frame underside to the upper frame top. B is the clearance from the ground to the lower frame underside. C is the upper frame top above the ground. The clear gap between the layers is one hundred and fifty millimetres. All these layout dimensions still need checking against the real components.

### 中文释义

上下两层都采用2525铝型材，外廓初拟800×450毫米。右图三种高度的起终点不同：A是框架自身总高，从下层框架底面到上层框架顶面，等于25＋150＋25＝200毫米；B是下层框架底面的离地间隙，初拟140毫米；C是上层框架顶面距地面的高度，为140＋200＝340毫米。150毫米是两层型材之间的净空，不是整车高度。轴距650毫米、轮距740毫米和轮径220毫米也都是待实物接口确认的初值。

### Delivery note · 讲述批注（不读）

**EN:** Identify the endpoints of A, B and C before giving their values.

**中文：** 先指出A、B、C尺寸线的起终点，再解释200、140、340毫米；不要混称整车高度。

---

## 04 · 双叉臂悬架与后轮传动

建议约40秒。

### English speech · 英文演讲正文

We use double-wishbone suspension at the front and rear. Each wheel has an upper wishbone, a lower wishbone and an inclined damper. The front wheels retain steering tie rods. The rear wheel carriers do not steer, but they still move with the suspension. Each rear motor drives its own wheel through a belt. The motor support and wheel bearing carrier must move together to maintain pulley spacing. The wheel is supported by its own bearings, rather than loading the motor shaft directly.

### 中文释义

前后统一称为双叉臂悬架，每轮有上叉臂、下叉臂和斜置减震器。前轮保留转向拉杆；“后轮不转向轮座”表示轮胎仍可旋转、轮座仍可随悬架运动，但轮座不能像前轮一样自由左右转向。它不等于把后轮刚性固定到车架，也不能只拆除前轮转向拉杆就认为后轮方向已受约束。两个后轮电机分别用皮带驱动对应车轮；电机支架与轮轴轴承座应共同随轮端运动以维持中心距。车轮由独立轴承支撑，不直接把轮载交给普通电机轴。

### Delivery note · 讲述批注（不读）

**EN:** Separate steering restraint from suspension movement.

**中文：** 指上下叉臂，再指后轮电机。强调不转向不等于没有悬架运动。

---

## 05 · 四爪从上向下夹取

建议约46秒。

### English speech · 英文演讲正文

Our gripper has four fingers and approaches the bowl from above. The top view shows the four contact directions. After locating the bowl, the open gripper descends, the fingers close around the bowl, and a small test lift checks the grip. They do not slide underneath to support the bowl. During transport, the arm keeps holding it in a compact pose. At the destination, the arm lowers the bowl and releases only after placement. The bowl and water weigh about six hundred and fifty grams; the gripper adds to the arm's end load.

### 中文释义

我们的夹爪是四爪，从上向下接近碗。俯视图明确表示四个接触方向：视觉定位后，张开的四爪下降，围绕碗收拢，再小幅试提确认夹持；不是两爪从下面托起，也没有底部托盘。运输途中机械臂持续夹持，尽量采用紧凑姿态。到终点后下降放置，确认碗落稳才释放。碗直径约156毫米，碗和水约650克，机械臂负载还要加入夹爪及连接件。图为功能示意，爪型、接触位置和夹紧力待团队结构图与实测确定。

### Delivery note · 讲述批注（不读）

**EN:** Count all four fingers in plan view, then show the downward approach.

**中文：** 先数俯视图四个爪，再沿侧视图向下指；强调从上方夹取、全程保持。

---

## 06 · 控制架构与STM32

建议约39秒。

### English speech · 英文演讲正文

The proposed architecture gives the STM32 one common coordination role. Orin processes camera data, identifies the goal marker and plans the path and mission actions. These commands go through the STM32 to the rear motor drivers, steering servo and arm controller. The arm controller still drives its own joints. Wheel, arm and gripper state return through the feedback path. We must confirm that the selected arm controller accepts a suitable communication interface. Power distribution is separate from these control signals.

### 中文释义

本版拟采用统一协调链路：Orin处理相机数据、识别目标标记物、规划路径和任务动作；命令经过STM32协调，再送到后轮电机驱动器、转向舵机和机械臂控制器。STM32不替代机械臂内部的关节驱动。轮速、机械臂、夹爪状态与故障沿反馈链路返回。关于“机械臂是否必须经过STM32”：并不是所有机械臂都必须这样连接，但我们可以把这条链路作为本项目的统一控制方案；最终要确认所购机械臂支持的串口、CAN或其他协议，必要时适配。具体协议尚未选定，不能宣称已接通。配电与控制信号分开。

### Delivery note · 讲述批注（不读）

**EN:** Trace commands first, then feedback; distinguish coordination from joint driving.

**中文：** 从左往右讲指令，再讲虚线反馈。说明STM32协调执行，机械臂控制器负责关节驱动，接口尚待确认。

---

## 07 · 标记物导航与自主避障

建议约44秒。

### English speech · 英文演讲正文

The main objective is to reach a recognised marker, not simply follow a boundary. First, vision recognises the selected marker and estimates its position. We then use depth information to identify free space and obstacles, and plan a path towards that goal. During motion, the system updates the scene and replans around obstacles while keeping enough clearance for the vehicle and bowl. Boundary detection and path following support this process. On arrival, vision switches to bowl or placement localisation. An uncertain goal or a blocked route triggers a stop.

### 中文释义

我们的重点是识别标记物，并规划通往该标记物的路径，实现自主避障，而不是单纯沿边界跟随。先识别选定的目标标记物并估计位置，再结合深度信息确定自由空间与障碍，规划到目标的路线。行驶中更新环境，绕障并重规划，净空要考虑车体、机械臂与碗的扫掠范围。边界识别用于约束可行驶区域，路径跟踪用于执行已经规划的路线，这两者是辅助环节。到达后切换到碗或放置区域定位。标记物类型和识别方式待确定；目标不可信或没有可行通道时停车，不能把示意路线当作已经验证的导航结果。

### Delivery note · 讲述批注（不读）

**EN:** Emphasise the goal marker and planned route, not boundary following.

**中文：** 重点指目标标记物和通往目标的曲线。不要把本页说成单纯循线或跟随。

---

## 08 · 任务状态与异常处理

建议约40秒。

### English speech · 英文演讲正文

The task progresses through approach, grasp, transport, placement and completion. The vehicle stays stationary during manipulation. Transport speed depends on free space, braking capability, obstacles, turns and ground conditions. The straight-line design target does not apply everywhere. The stop branch means that unreliable perception, an uncertain grip, lost commands or a driver fault can interrupt any active state. The vehicle stops, the conditions are rechecked, and only then may the task return to a valid state. Remote takeover is available when needed.

### 中文释义

任务依次经过接近、夹取、运输、放置和完成，取放时车辆静止。运输速度由自由空间、制动能力、障碍、弯道和路况共同约束；35公里每小时只是沿用的直线设计目标，不代表全程都按这个速度行驶。你批注的“Hold / controlled stop”是统一异常分支：无论当前在哪个运行状态，只要视觉不可靠、夹持不确定、指令丢失或驱动器报错，都应暂停行驶并采取预定处理。重新确认状态和安全条件后，才能返回合适的任务状态；不是通信一恢复就自行继续跑。“Hold”指暂停任务或保持状态，不是额外托住碗的装置。必要时遥控接管。

### Delivery note · 讲述批注（不读）

**EN:** Explain the stop-and-recheck condition; a returning signal is not a restart permission.

**中文：** 指异常分支说明暂停与复查；恢复通信本身不等于允许恢复运动。

---

## 09 · 验证计划

建议约43秒。

### English speech · 英文演讲正文

This page is our future test plan, not a report of completed tests. We first check a wheel module for belt alignment, slow-speed control, braking and temperature. Then we check suspension travel and steering clearance, followed by the loaded arm and four-finger grip. Next come marker localisation, goal-reaching and obstacle-avoidance trials at low speed. Finally, we test the complete mission and increase speed gradually. Each stage has a record, such as logs, video, water mass or a failure report, so that the next decision is based on evidence.

### 中文释义

这一页说明“接下来按什么顺序验证方案”，不是汇报已经完成的试验。先测试单轮模块的皮带对齐、慢速、制动和温度，再检查悬架全行程与转向间隙；然后检查机械臂带载和四爪夹持。接着做低速标记物定位、到达目标和避障，最后串联完整取运放任务，再逐步提速。右侧写的是每个测试应该留下的证据：日志、视频、温度、剩余水量或失败记录。这样导师能判断我们的验证路径是否合理。具体通过阈值仍需结合硬件额定参数与课程指标确定，不能把“计划测试”写成“已经通过”。

### Delivery note · 讲述批注（不读）

**EN:** Use future tense: these checks have not been reported as completed.

**中文：** 强调全部是未来验证计划，不是已经完成的实验结果。

---

## 10 · 下一阶段工作

建议约35秒。

### English speech · 英文演讲正文

The next step is to freeze the interfaces for the motors, the arm and the wheel modules. We need motor performance and driver data, the arm's working envelope and full end load, and confirmed mounting and steering geometry. These inputs allow detailed CAD and a first module test. The course budget remains ten thousand yuan, including delivery, spares and rework. The team drawings will replace the current concept references before final submission.

### 中文释义

下一步确定三类接口：电机性能与驱动信息、机械臂工作空间及完整末端负载、轮端安装和转向几何。有了这些依据，才能细化CAD并做首个模块验证。课程预算仍按一万元执行，包含运输、备件和返工。团队自己的结构图完成后会替换当前概念参考，再用于最终提交。本页是总体设计阶段的收束与下一步行动，不表示这些接口已全部核实。

### Delivery note · 讲述批注（不读）

**EN:** Finish with the three interface decisions and the course budget.

**中文：** 收束到三类待确定接口，预算只说课程一万元口径。

---

## 11 · 结束与问答

建议约18秒。

### English speech · 英文演讲正文

To conclude, AstraGO connects target localisation, top-down four-finger gripping, goal-directed navigation and placement. Our next step is to verify the interfaces and build the first module. Thank you for listening. We welcome your questions and suggestions.

### 中文释义

最后总结：AstraGO将目标定位、从上向下的四爪夹取、面向目标的自主导航与放置串联起来。下一步是核实接口并制作首个模块。感谢聆听，欢迎老师提问与提出建议。

### Delivery note · 讲述批注（不读）

**EN:** Pause after the summary, thank the audience and stay on this page for questions.

**中文：** 读完总结后停顿，再说Thank you；停留本页等待提问，不主动翻到备用页。

---

## 12 · 备用：速度计算

仅问答时使用。

### English speech · 英文演讲正文

A candidate two-hundred-and-twenty-millimetre tyre has a circumference of about zero point six nine one metres. At thirty-five kilometres per hour, the wheel needs approximately eight hundred and forty-four revolutions per minute. Motor speed equals wheel speed multiplied by the reduction ratio. This relationship does not prove that the motor has enough loaded speed or torque. We still need the motor ratings, feedback and braking information before selecting pulley sizes.

### 中文释义

候选轮胎外径220毫米，周长约0.691米；35公里每小时对应9.72米每秒，所需轮速约844转每分钟。电机转速＝轮速×减速比，减速比定义为轮轴带轮齿数除以电机带轮齿数。这里使用正常排版的公式文字，不再用代码变量样式。公式只表示运动学关系，不能证明电机带载转速、转矩、温升或制动能力满足要求，带轮齿数仍需电机参数后才能选定。

### Delivery note · 讲述批注（不读）

**EN:** Backup only: the number is required wheel speed, not measured motor performance.

**中文：** 只有老师问速度/传动比时才打开。844 rpm是所需轮速，不能当作电机已达到的结果。

---

## 13 · 备用：型材清单

仅问答时使用。

### English speech · 英文演讲正文

The initial cut list assumes that the crossmembers fit between the full-length longitudinal rails. Four eight-hundred-millimetre rails, six four-hundred-millimetre crossmembers and four one-hundred-and-fifty-millimetre posts give six point two metres of net profile. The crossmember length is four hundred and fifty minus two times twenty-five millimetres. Saw kerf, waste and extra reinforcement are not included. Confirm the joints and actual component interfaces before ordering.

### 中文释义

初步切料假设纵梁贯通，横梁夹在两根纵梁之间。四根800毫米纵梁、六根400毫米横梁与四根150毫米立柱，净长合计6.2米。横梁400毫米来自450减去两侧各25毫米。六根横梁包括两层各两根端横梁和一根安装横梁，尚不包含锯缝、损耗与额外加强件。连接件、实际接口和团队结构图确认后才能定稿下单。

### Delivery note · 讲述批注（不读）

**EN:** Backup only: net profile length excludes waste and additional reinforcement.

**中文：** 只有老师问型材尺寸或用量时才打开。6.2米是理论净长，不含损耗和追加加强件。

---

## 排练提示

1. 开场讲四个任务词：Locate、Grip、Navigate、Place。
2. 技术页指图讲逻辑，不连续念所有参数。
3. 明确四爪从上方夹取，视觉目标是标记物导航与避障。
4. 未验证的尺寸、接口与速度使用proposed、target、will test等表述。
5. 第11页读完结束语后停留，等待提问；两页备用不要主动读。
