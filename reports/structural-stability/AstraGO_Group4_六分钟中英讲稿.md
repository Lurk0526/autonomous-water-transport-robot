# AstraGO Vehicle Stability Analysis

**第四组｜六分钟中英对照讲稿**

对应定稿 PPT 的第 1–10 页。现场读英文即可，中文用于理解和排练。每页时间包含指图和翻页，斜体的动作提示不用念。

## 第 1 页　Vehicle Stability Analysis｜0:00–0:15

**English**

Good morning, everyone. We’re Group Four. Today we’ll look at how the AstraGO frame responds to three loads: longitudinal impact, side impact, and landing on its side.

**中文**

大家好，我们是第四组。今天我们来看 AstraGO 车架在三个工况下的响应：纵向碰撞、侧碰，以及侧翻后的侧向落地。

## 第 2 页　Vehicle objectives and analysis scope｜0:15–0:45

*指图：先指前部机械臂，再指车架与悬架的连接位置。*

**English**

AstraGO carries an open bowl that the robotic arm has to hold throughout the journey. The frame supports the equipment and transfers loads through the suspension mounts. The vehicle’s speed-test target is at least thirty kilometres per hour. For this structural analysis, we isolated the bare frame, including the rails, crossmembers, and brackets, and used separate load assumptions.

**中文**

AstraGO 运送的是一只敞口碗，机械臂需要在整个运输过程中夹住它。车架负责承托设备，并通过悬架安装点传递载荷。车辆的测速目标是至少每小时 30 公里。本次结构分析单独考察裸车架，包括纵梁、横梁和角码，并为各分析工况设定了对应的载荷。

## 第 3 页　Frame materials and their roles｜0:45–1:20

*指图：按纵梁、横梁、角码的顺序指右图，最后回到表中的设备板。*

**English**

The two long members are the sixty-by-sixty Q aluminium rails. They carry loads along the length of the frame. The three forty-five-by-forty-five Q crossmembers tie those rails together. The ADC twelve angle brackets connect the two sets of members. On the vehicle, a four-millimetre acrylic deck transfers equipment loads to the frame. In the simulation, we focus on the metal frame and its bracket connections.

**中文**

图中两根较长的是 6060Q 铝型材纵梁，负责沿车长方向传递载荷。三根 4545Q 横梁把两侧纵梁连在一起，ADC12 铝合金角码连接纵梁和横梁。实际车辆上，设备载荷通过 4 毫米厚的亚克力板传到车架。仿真主要考察金属车架及角码连接。

## 第 4 页　Three equivalent static load cases｜1:20–2:15

*指图：先指加速度列，再用手势区分纵向和侧向。*

**English**

We used SimScale to apply equivalent static loads to the bare frame. For longitudinal and side impact, we assumed an impact speed of two metres per second and a stopping distance of ten millimetres. With constant deceleration, speed squared divided by twice the stopping distance gives two hundred metres per second squared. We apply that acceleration along the relevant axis and include vertical gravity in both cases.

For rollover landing, the assumed drop height gives a different acceleration. All three cases use a linear elastic material model and sixteen bonded connection groups.

**中文**

我们用 SimScale 对裸车架施加等效静力载荷。纵向碰撞和侧碰都假设碰撞速度为每秒 2 米，停止距离为 10 毫米。在匀减速假设下，用速度的平方除以两倍停止距离，得到 200 米每二次方秒的等效加速度。两个工况分别沿对应方向施加加速度，并计入竖直重力。

侧翻落地的加速度根据假设落差计算。三个工况都采用线弹性材料模型，并使用 16 组绑定连接。

## 第 5 页　SolidWorks mesh and joint detail｜2:15–2:45

*指图：从左边整体网格移到右边角码的局部网格。*

**English**

On the left is the full frame mesh from SolidWorks. On the right, you can see the local mesh around one angle bracket, where load transfers between a rail and a crossmember. The SolidWorks mesh record shows element size settings from two to ten millimetres and about one point nine million elements. The response plots that follow are from SimScale.

**中文**

左边是 SolidWorks 中的整车架网格。右边放大了一个角码连接处，这里是载荷在纵梁和横梁之间传递的位置。SolidWorks 的网格记录中，单元尺寸设置为 2 到 10 毫米，约有 190 万个单元。后面的响应云图来自 SimScale。

## 第 6 页　Longitudinal loading｜2:45–3:30

*指图：按左、中、右依次展示固定位置、位移、应力。*

**English**

The first case represents front and rear loading on the bare frame under mirrored geometry, materials, joints, loads, and supports. The left image shows fixed support on the end faces of both rails. In the middle plot, displacement builds up along both rails away from that end. On the right, the von Mises stress shows higher bands along the rails and near several bracket interfaces. Each plot uses its own colour scale, so we focus here on the locations of higher stress and displacement.

**中文**

第一个工况代表裸车架的前后向受载，假设几何、材料、连接、载荷和支承均前后镜像。左图显示两根纵梁的端面被固定。中图中，位移沿两根纵梁远离固定端逐渐增大。右图的 von Mises 等效应力沿纵梁以及部分角码接口附近出现较高色带。每张云图使用独立色标，这里主要观察较高应力和位移出现的位置。

## 第 7 页　Side impact｜3:30–4:15

*指图：指出固定面，再沿三根横梁比划到对侧纵梁。*

**English**

For side impact, we fix two outer faces on the positive Y side. The lateral acceleration is two hundred metres per second squared, with gravity acting vertically. In the middle plot, the fixed rail moves very little. Displacement increases across all three crossmembers towards the opposite rail. The stress plot shows higher bands on the crossmembers and near several connections. Those regions are where we would look more closely at how the brackets transfer load between the rails.

**中文**

侧碰工况固定正 Y 侧的两个外表面。横向加速度是 200 米每二次方秒，同时计入竖直重力。中间的位移图显示，固定侧纵梁位移很小，位移沿三根横梁朝对侧纵梁增大。应力图中，横梁和部分连接附近出现较高应力色带。要进一步了解两侧纵梁之间如何通过角码传力，可以重点看这些区域。

## 第 8 页　Rollover landing｜4:15–5:00

*指图：指出侧面固定位置，再指对侧纵梁和横梁上的色带。*

**English**

For rollover landing, we model the equivalent load when the bare frame lands on its side. An assumed ten-centimetre drop gives a landing speed of about one point four metres per second. We use the same ten-millimetre stopping distance and add gravity in the landing direction. That gives one hundred and seven point nine one metres per second squared. We keep the same fixed faces as in the side-impact case. The displacement again increases across the frame, and the stress plot shows higher bands around the crossmembers and some of their joints.

**中文**

侧翻落地工况用等效载荷表示裸车架侧向着地时的受力。假设落差为 10 厘米，落地速度约为每秒 1.4 米。仍取 10 毫米的停止距离，再叠加与落地方向相同的重力，得到 107.91 米每二次方秒。固定面与侧碰工况相同。位移同样沿车宽方向增大，横梁和部分接头附近也能看到较高应力色带。

## 第 9 页　How the frame responds in each case｜5:00–5:50

*指图：先看表格的位移列，再看右侧应力区域列。*

**English**

The comparison shows two main displacement patterns. With end support, displacement increases along the longitudinal rails. With side support, it increases across the frame towards the opposite rail. In both side-supported cases, the stress contours draw attention to the crossmembers and their connections.

These results describe the bare frame under the assumed static loads and bonded connections. Each plot uses its own colour scale. Comparing peak stress or displacement requires the numerical maxima from SimScale. For the next structural check, we’d focus on the rail-to-crossmember joints and include the measured equipment loads.

**中文**

对照起来，可以看到两种主要的位移分布。端面固定时，位移沿纵梁方向增大。侧面固定时，位移沿车宽方向朝对侧纵梁增大。两组侧面固定工况的应力云图都提示我们关注横梁与连接区域。

这些结果反映的是裸车架在假设静力载荷和绑定连接条件下的响应。各图使用独立色标。要比较三个工况的应力或位移峰值，需要读取 SimScale 中的实际最大值。下一轮结构检查可重点复核纵梁和横梁的连接，并加入实测的设备载荷。

## 第 10 页　Thank you｜5:50–6:00

**English**

Thank you for listening. We’re happy to take any questions.

**中文**

谢谢大家，欢迎提问。
