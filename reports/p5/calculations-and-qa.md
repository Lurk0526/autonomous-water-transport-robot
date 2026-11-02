# AstraGO P5 V5：5分半中英讲稿

正文11页，英文稿约639词。按约125词/分钟并留出翻页、指图停顿，目标约5分半；以本人排练计时为准。英文用于演讲，中文用于理解。第12—13页为备用资料，不计入正文。

重点：车架计算减重、后置电池避免专用配重、按任务选择电池、悬挂与橡胶支撑。

## 01_cover

Good morning. We are Group Two. AstraGO transports an open water bowl through continuous robotic gripping. Our design combines lightweight construction with vibration control, focusing on material selection, frame geometry, load distribution, manufacturing and battery sizing.

大家好，我们是第二组。AstraGO通过机械臂持续夹持敞口水碗完成运输。我们的设计将轻量化与振动控制结合，重点介绍材料选择、车架结构、载荷分布、制造方法和电池容量。

---

## 02_materials

Each material serves a specific function. Aluminium profiles provide a low-density frame with reusable mounting slots. PLA enables low-cost custom fingers and supports our lightweight arm-end design. Steel is concentrated at the arm interfaces to retain the required connections. Acrylic makes the equipment deck convenient to cut and drill. This combination considers material, geometry and manufacturing together, giving each part a clear role in the vehicle.

每种材料承担明确功能。铝型材提供低密度车架和可重复使用的安装槽；PLA便于低成本制作定制夹指，服务于末端轻量化设计；钢件集中用于机械臂接口，保留所需连接；亚克力便于铺板切割和钻孔。这样的组合把材料、几何和制造方式一起考虑，让每个部件各司其职。

Sources: European Aluminium, https://european-aluminium.eu/wp-content/uploads/2022/11/aam-design-2-design-with-aluminium.pdf ; Prusa, https://help.prusa3d.com/article/pla_2062 .

---

## 03_structure

The frame provides our main calculated weight reduction. The original design used forty-five-millimetre profiles throughout. We removed the uprights and braces, reducing profile pieces from nineteen to five and brackets from eighteen to eight. Both longitudinal rails were shortened from eight hundred to six hundred and fifty millimetres, together with the deck. Calculated profiles plus acrylic decrease from seven point nine one to five point two nine kilograms, a saving of about two point six two kilograms, or thirty-three percent.

车架是主要的计算减重来源。旧版全部采用4545型材，我们取消立柱和斜撑，型材从19件减少到5件，角码从18个减少到8个。两根纵梁由800毫米缩短到650毫米，铺板同步缩短。型材与亚克力的计算质量由7.91公斤降到5.29公斤，减少约2.62公斤、约33%。

Calculation scope: aluminium profiles + acrylic deck. Supplier linear masses: 4545 1.50 kg/m, 6060 2.38 kg/m. Acrylic reference density: 1190 kg/m³.

---

## 04_layout

The battery also contributes through its position. The front arm and suspended bowl place load towards the front, so we position the battery at the rear to help balance them. One necessary component then performs two functions: energy storage and load balance. This is our approach to avoiding a separate ballast block and the extra mass added solely for balance.

电池的位置也参与轻量化设计。前端机械臂和悬空水碗使载荷趋向前部，因此把电池放在后端参与平衡。这样一个必需部件同时承担储能和配重两个功能，目标是避免专门增加配重块，省去只为平衡而增加的质量。

---

## 05_printing

Three-dimensional printing suits the small batch and custom finger and linkage geometry. The photograph shows the gripper we manufactured. We adapted the mounting interface of the supplier-based mechanism to connect it to our arm. Solid PLA forms the finger structure, while rubber pads provide the bowl-contact surfaces. The process makes these custom shapes practical to produce and modify.

3D打印适合这次小批量夹指和连杆的定制形状。照片展示了已经制作的夹爪，我们在商家机构基础上适配安装接口，使其与机械臂连接。实心PLA形成夹指结构，橡胶垫提供与碗接触的表面。打印使这些定制形状便于制造和修改。

---

## 06_battery

We select a forty-eight-volt, twenty-amp-hour pack for our design. The two motors reach a combined two-kilowatt peak only during acceleration. We therefore distinguish short power demands from the energy required over the task. The selected pack offers thirty amps continuous and a specified sixty-amp peak, with nine hundred and sixty nominal watt-hours. A forty-amp-hour pack doubles that stored energy. Our sizing approach uses the smaller capacity within its operating envelope, avoiding capacity beyond the task requirement.

本次设计选择48伏20安时电池。两个电机合计两千瓦是加速时的短时峰值，因此选型时区分瞬时功率与整个任务的能量需求。所选电池持续放电30安、标示峰值60安，标称储能960瓦时；40安时则使储能翻倍。我们的选型思路是在工作范围内采用较小容量，避免超出任务需求的容量配置。

Design envelope: total continuous current ≤30 A; acceleration peaks within the supplier current-duration rating; usable pack energy covers task demand. Slide calculation uses 2 kW traction electrical input at 48 V. Source: supplier nameplates supplied by the team.

---

## 07_disturbance

We control disturbance along its transmission path: from the road, through the wheels and suspension, to the frame, arm and bowl. Suspension responds to bumps, deck pads provide compliance, and rubber jaw pads support stable contact. Smooth starts and stops complement these mechanical measures.

我们沿扰动传递路径布置措施：从路面到车轮与悬挂，再到车架、机械臂和水碗。悬挂响应颠簸，铺板垫块提供柔性，夹爪橡胶垫保持接触稳定，并通过平缓起停配合这些机械措施。

---

## 08_suspension

Our preliminary spring selection is six newtons per millimetre at each wheel. Using a design mass of fifteen to twenty kilograms, fifty millimetres of wheel travel and an assumed motion ratio of zero point seven, calculated static sag is thirteen to seventeen millimetres. This uses about twenty-five to thirty-three percent of the travel, leaving upward movement for bumps. The spring provides support and compliance, while damping controls rebound, balancing rough-ground response with carrying stability.

四轮弹簧初选为每根6牛每毫米。按15至20公斤设计质量、50毫米轮心行程及0.7的运动比假设，计算静态下沉约13至17毫米，占总行程约25%至33%，保留应对颠簸的上跳空间。弹簧提供支撑和柔性，阻尼控制回弹，兼顾越野响应和持碗稳定性。

Calculation model: r = spring displacement / wheel displacement = 0.7 assumed; k_wheel = 6 × 0.7² = 2.94 N/mm. Equal corner loads; total design mass approximated as sprung mass; zero preload; locally constant motion ratio. Sag = mg/(4 k_wheel). Reference: https://www.hypercoils.com/spring-rate-calculator/ .

---

## 09_rubber

Our deck-support design uses eight rubber pads: four at the rail corners and four on the crossmembers. The two sizes match these different interfaces. The pads form a compliant layer between the frame and equipment deck. Their placement combines distributed support with the aim of reducing transmitted disturbance, while the jaw pads address bowl contact.

设备板支撑方案采用八块橡胶垫：纵梁四角四块、横梁四处四块，两种尺寸分别匹配不同接口。垫块在车架与设备板之间形成柔性层，通过分布支撑减少传递扰动；夹爪垫片则处理与碗的接触。

---

## 10_precedent

A recent Fraunhofer and MOSOLF vehicle shelf demonstrates how integrated printed geometry can reduce additional attachments. We apply this principle selectively: custom geometry and mounting functions should be considered together. For our small-batch gripper, accessible PLA printing and an adapted interface fit the project scale and fabrication needs.

Fraunhofer与MOSOLF的汽车置物架案例说明，集成打印几何可以减少额外安装件。我们有选择地借鉴这一原则，将定制形状与安装功能一起考虑。对于小批量夹爪，可实现的PLA打印和接口适配符合项目规模与制造需求。

Source: Fraunhofer IWU, 12 November 2025, https://www.iwu.fraunhofer.de/en/press/2025-fraunhofer-iwu-experts-present-new-materials-for-additive-manufacturing-at-their-self-printed-booth.html .

---

## 11_conclusion

Our design combines a shorter frame, functional material choices, balanced equipment placement, custom printing and appropriate battery capacity. The profiles and deck achieve about thirty-three percent calculated mass reduction. Suspension and compliant interfaces complete our approach to stable water-bowl transport. Thank you.

我们的设计结合了缩短车架、按功能选材、平衡设备布局、定制打印和适当电池容量。型材与铺板的计算减重约33%，悬挂和柔性接口共同服务于稳定运水。谢谢。

---

## 12_mass_backup

The calculation uses supplier linear masses and the acrylic panel dimensions. Old profiles total four point six five one three six metres at one point five kilograms per metre. New profiles combine one point three metres of sixty-millimetre rail with zero point seven two metres of forty-five-millimetre crossmember. Including the acrylic panels gives seven point nine one zero and five point two eight eight kilograms, a reduction of two point six two two kilograms, or thirty-three point one five percent.

计算采用商家米重与亚克力板尺寸。旧版4545型材总长4.65136米，米重1.50公斤；新版由1.30米6060纵梁和0.72米4545横梁组成，米重分别为2.38和1.50公斤。亚克力按1190公斤每立方米计算，两项小计分别为7.910和5.288公斤，减少2.622公斤、33.15%。比较范围为型材和亚克力。

Old length inventory: 840×1 + 760×2 + 310×4 + 140×4 + 61.42×8 mm. Old deck: 2×560×175×4 mm. New profiles: 2×650 mm 6060 + 3×240 mm 4545. New deck: 650×360×4 mm.

---

## 13_qa_backup

Our choices follow component function and task demand. Frame geometry controls material use. Battery placement combines power supply with balancing, while capacity selection separates energy needs from acceleration peaks. PLA printing serves the custom gripper geometry, with steel used at the arm interface. Spring selection uses a defined mass, travel and motion-ratio model; rubber pads provide compliant interfaces along the disturbance path.

设计选择围绕部件功能和任务需求展开。车架几何控制材料用量；电池布局兼顾供能和平衡，容量选择则区分能量需求与加速峰值。PLA打印实现夹爪定制形状，钢件用于机械臂接口。弹簧初选采用明确的质量、行程和运动比模型；橡胶垫在扰动路径上提供柔性接口。
