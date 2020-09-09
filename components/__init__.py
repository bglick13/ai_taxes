"""
Components contain logic that makes part of the environment run. For example, resource trading and bracketed income tax
are built in components for Foundation. Here, we can implement our custom components for:

- Managing citizenship
- Multi-planner taxing (to accommodate for multiple "nations")
"""

from components.citizenship import *
from components.multi_nation_bracketed_income_tax import *