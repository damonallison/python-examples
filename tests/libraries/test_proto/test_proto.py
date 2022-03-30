from typing import Type, TypeVar

from tests.libraries.test_proto.compiled import messages_pb2
from google.protobuf.message import Message


ProtoMessage = TypeVar("ProtoMessage", bound=Message)


def serialize(msg: Message) -> bytes:
    return msg.SerializeToString()


def deserialize(b: bytes, t: Type[ProtoMessage]) -> ProtoMessage:
    obj: ProtoMessage = t()
    assert isinstance(obj, Message)
    obj.ParseFromString(b)
    return obj


def test_proto_serdes():
    p = messages_pb2.Person()
    p.name = "damon allison"
    b = serialize(p)
    p2 = deserialize(b, messages_pb2.Person)
    assert p2.name == "damon allison"
