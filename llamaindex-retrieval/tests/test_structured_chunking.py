from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from llamaindex_retrieval.structured_chunking import structure_aware_nodes


def document(text: str) -> Document:
    return Document(
        id_="metro-document",
        text=text,
        metadata={
            "document_id": "metro-document",
            "file_id": "metro-document",
            "title": "测试线路",
            "source": "test",
            "kind": "finewiki",
        },
    )


def test_table_is_split_on_rows_and_has_complete_first_column_summary() -> None:
    stations = [f"车站{index}" for index in range(1, 31)]
    rows = "\n".join(f"| {station} | 区域{index} |" for index, station in enumerate(stations))
    source = f"""# 测试线路

## 车站
全线共有30座车站。
| 站名 | 所在地 |
| --- | --- |
{rows}
| 注释 | 不属于数据行 |

## 历史
2020年开始运营。
"""

    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )

    table_nodes = [node for node in nodes if node.metadata["content_type"].startswith("table")]
    summaries = [node for node in table_nodes if node.metadata["content_type"] == "table_summary"]
    assert len(summaries) == 1
    assert all(station in summaries[0].text for station in stations)
    assert "注释" not in summaries[0].text
    assert summaries[0].metadata["section"] == "测试线路 > 车站"
    assert len({node.metadata["parent_id"] for node in table_nodes}) == 1
    assert [node.metadata["chunk_order"] for node in table_nodes] == list(
        range(len(table_nodes))
    )
    assert all("站名：" in node.text for node in table_nodes[1:])


def test_table_cells_drop_wiki_reference_marks() -> None:
    source = """# 测试线路

## 车站
| 站名 | 所在地 |
| --- | --- |
| 后瑞[4] | 宝安 |
| 机场东[12] | 宝安 |
"""

    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )

    combined = "\n".join(node.text for node in nodes)
    assert "后瑞[4]" not in combined
    assert "机场东[12]" not in combined
    assert "站名列表：后瑞、机场东" in combined


def test_table_summary_repairs_single_character_station_from_document_context() -> None:
    source = """# 测试线路

正文说明：除机场东站和后瑞站为高架车站。

## 车站
| 站名 | 所在地 |
| --- | --- |
| 固戍 | 宝安 |
| 瑞 | 宝安 |
| 机场东 | 宝安 |
"""

    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )

    combined = "\n".join(node.text for node in nodes)
    assert "站名列表：固戍、后瑞、机场东" in combined
    assert "站名：后瑞" in combined


def test_lists_key_values_and_prose_get_generic_structure_metadata() -> None:
    source = """# 使用说明

普通介绍文字。

## 操作步骤
- 打开页面
- 选择文件
- 点击导入

## 规格
型号：A1
容量：10GB
"""
    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )
    types = {node.metadata["content_type"] for node in nodes}
    assert {"prose", "list", "key_value"} <= types
    assert all(node.metadata["parent_id"] for node in nodes)
    assert all("section" in node.metadata for node in nodes)


def test_prose_with_colons_is_not_misclassified_as_key_values() -> None:
    source = """# 历史

官方曾经发布两种方案：第一种方案保留原名称，第二种方案采用新名称。
后来经过公开讨论：主管部门最终确认采用第二种方案，并公布实施日期。
"""
    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )
    assert {node.metadata["content_type"] for node in nodes} == {"prose"}


def test_document_aliases_are_attached_to_every_chunk() -> None:
    source = """# 深圳地铁1号线

深圳地铁1号线，简称罗宝线，是深圳地铁的一条线路。

## 车站
- 罗湖站
- 世界之窗站
"""

    nodes = structure_aware_nodes(
        document(source),
        SentenceSplitter(chunk_size=128, chunk_overlap=16),
    )

    assert nodes
    assert all("罗宝线" in node.metadata["aliases"] for node in nodes)
