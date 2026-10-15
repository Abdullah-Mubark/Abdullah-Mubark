"""Model adapters: one small interface for every detector / recogniser."""

from .base import BaseAdapter, Box, PlateRead
from .registry import create_adapter, register

__all__ = ["BaseAdapter", "Box", "PlateRead", "create_adapter", "register"]
