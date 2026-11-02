# 来源、归属与文件索引

本次以已有项目资料为事实依据，结合记忆定位历史上下文，再以工作区较新的 P5 确认覆盖旧状态。未将申请材料中的职责或未来计划写成已完成的系统成果。

## 原始文件与历史依据

| 来源 | 仓库位置 | 解释范围 |
| --- | --- | --- |
| 2026-11-02 用户提供的整车总装 | [CAD v2](../hardware/cad/v2/README.md) | 文件已接收；未确认引用齐全或实际几何 |
| 同次提供的机械臂配件 ZIP | [CAD 清单](../hardware/cad/v2/manifest.json) | CRC 检查通过；32 个文件解压，原包和文件名保留 |
| P4 事实基础，2026-10-19 | [P4 source](sources/p4/content_basis.md) | 视觉跟踪、电机独立控制、采购与安装状态 |
| P5 较早事实基础，2026-10-24 | [P5 V2 source](sources/p5/content_basis_v2.md) | 车架已装配、布置与材料；其尺寸/质量已被后续修订覆盖 |
| P5 用户直接更新，2026-10-24 | [用户更新](sources/p5/user_update.md) | 橡胶垫尺寸、实心打印、夹具质量为估计 |
| P5 较新修订，2026-10-25 | [V4 source](sources/p5/revision_v4_current_basis.md) | 650 mm 纵梁、质量口径、电池倾向和弹簧初选 |
| 质量重算确认 | [计算输入](sources/p5/frame_mass_recalculation_confirmed.md) | 型材与铺板同范围的计算估算 |
| 悬架初选 | [弹簧说明](sources/p5/suspension_preliminary_choice_6Nmm.md) | 6 N/mm 为假设条件下的初选，不是实物测量 |
| P1–P5 选定汇报与讲稿 | [报告索引](../reports/README.md) | 保存当时的设计表达，不自动代表当前完成状态 |
| 台架、夹具照片及 CAD 参考图 | [图像索引](../assets/images/README.md) | 外观/布置证据；图片不能独立证明动态性能 |

原工作区 `PROJECT_CONTEXT.md` 更新至 2026-10-19，作为查阅入口使用；其中保留大量更早方案，未整体复制入仓库。原工作区的 PPT 构建缓存、全部中间版本、临时文件和演示制作脚本也不纳入机器人运行代码。

## 完整性

[import-manifest.json](import-manifest.json) 记录逐项来源标签、目标路径、文件大小和 SHA-256。原件、选定图像、PPTX 与讲稿按字节复制，不在归档时改写内容；PPTX 做 ZIP 完整性检查，不因此声称重新验证了页面布局。

[CAD manifest](../hardware/cad/v2/manifest.json) 另记录 ZIP 中文名解码、CRC 与解压文件哈希。哈希证明文件一致性，不证明设计正确、依赖完整或引用合法。

## 归属与许可

四爪基础设计、部分机械臂/轮端 CAD、供应商参数与报告中的外部图片不应一概归为团队原创。团队已明确的改造是夹具与机械臂末端安装接口；具体 CAD 文件的创作者和许可尚未逐个确认。

本次创建私有仓库，未为全部材料套用 MIT 等开源许可证。未来若公开或对外再分发，按来源区分团队原创、供应商资料和第三方素材，并核对对应授权。此说明不替代各文件已有许可。

## 外部依据

- [SolidWorks：Pack and Go](https://help.solidworks.com/2022/english/SolidWorks/sldworks/c_Pack_Go_Ovw_WPDM.htm)：模型相关文件的打包方式。
- [GitHub：普通 Git 的大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)：大文件上传规划。
