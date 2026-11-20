# mesh-winding-repair

## 题面

导入外部模型后，一半的面法线朝内、开灯就全黑：法线用 `(p2-p0)×(p1-p0)`，整体反了；有符号体积不除 6，数值差 6 倍；翻转只换前两个顶点，等价于没翻；流形判定允许一条边 3 个面，非流形漏检；BFS 不传递翻转，相邻面朝向相反；多连通分量只处理第一个，其余保持原样；闭合体判定用包围盒，开放面也被判闭合；修复不剔退化面，体积计算为 0；批量共用翻转标记，网格之间串；空网格不校验，除零求体积。winding/repair.py 是核心，tests/test_repair.py 15个红。要修的是：按右手系重算法线并归一化、有符号体积除以 6、翻转交换后两个顶点、按每条边恰好两个面判定流形、BFS 向相邻面传递翻转、遍历全部连通分量、按边计数而非包围盒判定闭合、修复时剔除退化面、批量隔离每个网格的翻转标记、空网格先校验并抛 `WindingError`。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态应恰好为 `15 failed`，且没有 collection error、import error 或 syntax error：

```text
15 failed in 0.XXs
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定

- 只使用 Python 标准库 `math`、`collections` 与 `pytest`，离线运行，不使用随机数。
- 右手系：三角形从正面看为逆时针，外向法线为 `normalize((p1-p0)×(p2-p0))`。
- 闭合、朝向一致的网格有符号体积为正，表示法线朝外。
- 浮点比较容差为 `1e-6`。
- 网格字段固定为 `{"positions", "faces"}`，`faces` 为三角形下标三元组。
- 自定义异常类为 `winding.repair.WindingError`。

## 不可修改

`tests/test_repair.py` 不可修改。修复只能发生在 `winding/repair.py`，并保持现有公开函数签名、返回结构与 docstring 契约不变。
