# AstraGO P4：7–8 分钟中英对照讲稿

英文讲述，中文用于理解与排练；各页时间仅为排练分配。若中英文都读出会超过目标时长。
英文约 897 词；目标约 7 分 30 秒，请按实际语速排练。


## 1. Control System & Customised Components（约 20 秒）

**English**

Good morning. This is our fourth presentation on AstraGO. We will explain the control architecture, our progress in vision-guided arm tracking and drive motor control, and the latest chassis and gripper adaptations. We will also distinguish the work already demonstrated from the next integration tasks.

**中文对照**

大家好，这是 AstraGO 的第四次汇报。我们将介绍控制架构、机械臂视觉跟踪和驱动电机控制的进展，以及最新的车架与夹爪适配工作，同时说明哪些功能已经验证、哪些还需要集成。

## 2. Progress Since P3（约 35 秒）

**English**

Since P3, we have moved from component selection towards bench testing and fabrication. The arm can now follow a moving black bowl using our vision pipeline. We have also tested individual drive motor commands through STM32. The revised frame is being fabricated and assembled. The four-finger gripper has been manufactured, and the next step is to mount it using the adapted interface. These developments form the basis for our next stage of integrated testing.

**中文对照**

相比 P3，我们已从部件选型推进到台架调试与加工。机械臂能够通过视觉链路跟随移动的黑色圆碗，驱动电机也已通过 STM32 完成独立指令调试。新版车架正在加工装配，四爪已经加工，下一步通过适配接口装到机械臂上。这些进展为后续集成测试建立了基础。

## 3. Control System Architecture（约 60 秒）

**English**

This diagram presents the functional architecture of our vehicle. Orin Nano receives camera information and handles visual perception, planning and task coordination. It sends motion targets to STM32, which is the common command interface for the drive controller, steering servo and arm servos. There is no additional chassis control board. Feedback from the actuators provides information for monitoring and subsequent control calculations. Manual remote emergency power-off is included in the system design. The arm tracking chain and local drive motor commands are the parts demonstrated in our current bench work. The following slides explain these functions and the mechanical components that support them.

**中文对照**

这张图介绍车辆的功能架构。Orin Nano 接收相机信息，负责视觉感知、规划和任务协调，再将运动目标发送给 STM32。STM32 统一向电机控制器、转向舵机和机械臂舵机下发指令，不额外设置一层底盘控制板。执行器反馈为状态监测和后续控制计算提供信息，人工远程紧急下电也纳入系统设计。当前台架工作已演示机械臂跟踪链路和驱动电机的局部控制，后续页面将介绍这些功能及其对应的机械部件。

## 4. Mission Sequence & Driving Strategy（约 50 秒）

**English**

The intended mission starts with the vehicle approaching the pickup station and stopping before the arm operates. After locating the bowl, the gripper closes and performs a small test lift. The arm then moves into a compact carrying posture and holds the bowl throughout transport, without a supporting tray. Driving speed depends on the section of the course. Concrete, gravel, brick surfaces, obstacles and turns require controlled low-speed movement. Thirty kilometres per hour is the target at the clear speed checkpoint, rather than a constant speed everywhere. Acceleration and braking space still need checking. Autonomous navigation and the complete pickup-to-placement sequence have not yet been demonstrated.

**中文对照**

目标流程从车辆接近取碗台并停车开始。定位碗后夹紧并小幅试提，再收拢到紧凑运输姿态，全程持续夹持，不使用碗托承重。车速随路段变化：水泥、砂石、砖块混合路面，以及障碍和弯道，需要受控低速通过。30 公里每小时是无遮挡测速点的目标，不是全程速度。加速和制动空间还需核对。自主导航和从取碗到放碗的完整任务目前尚未演示。

## 5. Bowl Segmentation with YOLO11n-seg（约 50 秒）

**English**

For bowl perception, we fine-tuned YOLO11n-seg using more than five hundred images collected by our team. The images cover different target positions, lighting conditions and backgrounds. We randomly divided them into eighty percent training, ten percent validation and ten percent testing. The trained model is used by Orin in our live tracking pipeline. Our practical target is the black round bowl used in the task. The segmentation result supports tracking commands, so that the arm can follow the bowl as its position changes. This connects model development to the bench demonstration shown on the next slide.

**中文对照**

碗识别采用团队微调的 YOLO11n-seg，使用自采的 500 多张图片，覆盖不同目标位置、光照和背景，并按训练、验证、测试 8∶1∶1 随机划分。训练后的模型用于 Orin 上的实时跟踪链路，实际目标是任务中的黑色圆碗。分割结果支持跟踪指令计算，使机械臂能够随碗的位置变化而运动。下一页将展示模型开发与台架调试之间的联系。

## 6. Vision-Guided Arm Tracking（约 70 秒）

**English**

These photographs show our bench setup, with the arm, OpenMV module and Orin Nano connected for visual tracking. OpenMV supplies images. Orin runs the trained segmentation model and calculates tracking commands, and STM32 sends the commands to the arm servos. When the black bowl moves, the arm follows so that the camera continues pointing towards it. This is the behaviour we have demonstrated. The visible preview in the photo documents image acquisition, while the tracking behaviour is reported from our bench tests. Tracking does not yet establish a calibrated three-dimensional grasp position. The next steps are calibration, approach motion and grasp confirmation with the new gripper. We will then test whether the bowl can be lifted and held reliably, rather than treating visual following as a completed grasp.

**中文对照**

照片展示了机械臂、OpenMV 和 Orin Nano 的台架调试布置。OpenMV 提供图像，Orin 运行训练后的分割模型并计算跟踪指令，STM32 再向机械臂舵机发送命令。当黑色圆碗移动时，机械臂随动，使相机继续对准目标，这是目前已演示的行为。照片中的预览记录了图像采集，跟踪行为来自团队台架测试。视觉跟踪还不代表已取得经过标定的三维抓取位置；下一步需要标定、接近动作和新夹爪的抓取确认，再验证碗能否被可靠提起和保持。

## 7. Drive Motor Control & Feedback（约 50 秒）

**English**

The drive motor has been tested separately through STM32 and its motor controller. We can start and stop the motor, change between forward and reverse, adjust the speed command and read actual speed feedback. This gives us a foundation for later vehicle control. However, receiving speed data is not by itself proof that a closed-loop speed controller has been implemented. We also have not yet completed the Orin-to-drive command chain or steering integration. Our next motor-related task is to bring up the steering servo, then coordinate steering and drive at low speed before attempting whole-vehicle tests.

**中文对照**

驱动电机已通过 STM32 和电机控制器完成单独调试，实现启停、正反转、调节速度指令，并读取实际转速反馈。这为后续整车控制建立了基础。但能读取转速，不等于速度闭环已经实现；Orin 到底盘的指令链和转向集成也还未完成。接下来先调通转向舵机，再进行低速驱动与转向协调，最后开展整车测试。

## 8. Customised Chassis Frame（约 50 秒）

**English**

The two views compare our previous P3 frame with the updated P4 design. The latest frame uses two sixty-by-sixty longitudinal rails and three forty-five-by-forty-five crossmembers. We changed the main profile arrangement and reduced unnecessary connecting parts. The earlier design used more supporting members around the wheel modules. In the new layout, the main frame provides a larger area for placing the arm, battery and Orin. The comparison shows our design direction: simpler connections and more useful equipment space. The materials and parts are available, and fabrication and assembly are underway. Our next step is to bring the frame and control hardware together.

**中文对照**

两张图对比了 P3 旧车架和 P4 更新设计。最新版采用两根 6060 纵梁、三根 4545 横梁，调整了主要型材布局并减少冗余连接件。旧版在轮端周围使用了更多支撑构件；新版主框架为机械臂、电池和 Orin 留出更大的安装区域。这一对比体现了简化连接、扩大有效设备空间的设计方向。材料与零件已经备齐，正在加工装配，下一步将车架与控制硬件集成。

## 9. Four-Finger Gripper Adaptation（约 45 秒）

**English**

Here we compare the original two-finger attachment on the left with the four-finger configuration on the right. The four-finger image is the supplier base model. Our team adapted its mounting interface so that the gripper can connect to the existing arm. The adapted parts have been manufactured, and the next step is installation and opening-and-closing tests. The four-finger layout is intended to make contact around the bowl. Its actual holding behaviour will depend on contact geometry, grip force and the added mass at the arm tip. We will check the assembled mechanism before progressing from light objects to the bowl and target load.

**中文对照**

左图是原装两爪末端，右图是四爪构型，四爪图片来自商家基础模型。团队修改了安装接口，使其能够连接现有机械臂。适配件已经加工，下一步是安装并测试开合。四指布局的目标是在碗的周向形成接触，实际夹持表现取决于接触几何、夹紧力和末端新增质量。我们会先检查装配后的机构，再从轻物逐步过渡到碗和目标载荷。

## 10. Next Milestones（约 20 秒）

**English**

Our next priorities are to mount the four-finger gripper and test its motion, integrate steering with drive control, and complete the chassis assembly. These steps will prepare the system for coordinated bench tests and later low-speed vehicle trials. Thank you. We welcome your questions.

**中文对照**

下一阶段优先安装四爪并测试开合、完成转向与驱动联调，以及车架装配，为协调台架测试和后续低速整车试验做准备。谢谢，欢迎提问。
