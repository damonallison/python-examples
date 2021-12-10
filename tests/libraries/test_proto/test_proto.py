from typing import Type, TypeVar

from tests.libraries.test_proto.compiled import messages_pb2
from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict


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


def test_proto_serdes_initial() -> None:
    """Proto objects are defined with metaclasses.

    These metaclasses use the descriptors to "create" the true class.

    If you assign a field to the incorrect type, a `TypeError` is raised.
    """

    person = messages_pb2.Person()
    person.id = 1234
    person.name = "Damon Allison"
    phone = person.phones.add()
    phone.number = "123-4567"
    phone.type = messages_pb2.Person.PhoneType.HOME

    # IsInitialized checks if all required fields have been set
    assert phone.IsInitialized()

    # ser
    s = person.SerializeToString()
    # print(s)

    # des
    person2 = messages_pb2.Person()
    person2.ParseFromString(s)

    assert person == person2

    print(MessageToDict(person))

    # Determine if a field exists
    assert person2.HasField("name")

    person3 = messages_pb2.Person()
    assert not person3.HasField("name")
