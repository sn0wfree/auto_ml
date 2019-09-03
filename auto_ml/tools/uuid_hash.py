# coding=utf8
import uuid


def uuid_hash(strings, uuid_type='uuid3', namespace=uuid.NAMESPACE_DNS):
    return getattr(uuid, uuid_type)(namespace, strings).hex
