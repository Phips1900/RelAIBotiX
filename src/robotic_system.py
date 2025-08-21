from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class Component:
    name: str
    failure_prob_per_min: float = 0.0  # base p0/min from JSON
    redundancy: bool = False
    properties: Dict[str, Dict[str, float]] = field(default_factory=dict)  # prop -> {skill: value}
    skills: List[str] = field(default_factory=list)

    def add_property(self, prop: str, value: float, skill: str) -> None:
        self.properties.setdefault(prop, {})[skill] = value

@dataclass
class Skill:
    name: str
    id: Optional[int] = None
    components: List[str] = field(default_factory=list)  # component names
    failure_prob: float = 0.0  # FT top-event (per exposure analyzed)

@dataclass
class RoboticSystem:
    name: str
    robot_type: str = ""
    components: Dict[str, Component] = field(default_factory=dict)  # name -> Component
    skills: Dict[str, Skill] = field(default_factory=dict)          # name -> Skill
    system_failure_prob: float = 0.0

    # Attach analyzer artifacts (audit-friendly)
    runs_df: Optional[pd.DataFrame] = None
    segments_df: Optional[pd.DataFrame] = None
    components_df: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, pd.DataFrame]] = None
    failure_table: Optional[pd.DataFrame] = None   # skill × component p_fail (already time/velocity-weighted)

    def add_component(self, c: Component) -> None:
        self.components[c.name] = c

    def add_skill(self, s: Skill) -> None:
        self.skills[s.name] = s

    def add_component_to_skill(self, comp_name: str, skill_name: str) -> None:
        self.skills.setdefault(skill_name, Skill(skill_name))
        if comp_name not in self.skills[skill_name].components:
            self.skills[skill_name].components.append(comp_name)
        self.components.setdefault(comp_name, Component(comp_name))
        if skill_name not in self.components[comp_name].skills:
            self.components[comp_name].skills.append(skill_name)