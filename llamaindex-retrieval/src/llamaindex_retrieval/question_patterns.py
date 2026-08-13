import re


_PERSON_QUESTION = r"(?:谁|哪几个人|哪些人|哪几位|哪位|什么人)"
_AGENT_ACTION = (
    r"(?:发起|提出|创立|创建|建立|发明|发现|开启|开辟|开通|"
    r"领导|指挥|主演|导演|执导|撰写|创作|设计|建造|开发|制作|主持|组织|推动|负责|"
    r"创始人|创办人|创办者|建立者|创建者|发明者|发现者|创作者|作者|设计者|建造者|执导者|"
    r"創作者|創辦者|設計者|執導者)"
)
_REVERSE_AGENT_RELATION = (
    r"(?:开国皇帝|创始人|创办人|创办者|建立者|创建者|发明者|发现者|创作者|作者|导演|主演|设计者|执导者|负责人|"
    r"得主|获得者|总统|副总统|总理|首相|主席|建造者|創作者|創辦者|設計者|執導者)"
)
AGENT_QUESTION_PATTERN = re.compile(
    rf"(?:由\s*)?{_PERSON_QUESTION}[^，。？?]{{0,12}}{_AGENT_ACTION}"
    rf"|{_AGENT_ACTION}[^，。？?]{{0,40}}?(?:的是|者是|的人是){_PERSON_QUESTION}"
    rf"|{_REVERSE_AGENT_RELATION}(?:是|为)?{_PERSON_QUESTION}"
)


def is_agent_relation_question(question: str) -> bool:
    """Return whether a question asks which person performed an action."""

    return bool(AGENT_QUESTION_PATTERN.search(question))
