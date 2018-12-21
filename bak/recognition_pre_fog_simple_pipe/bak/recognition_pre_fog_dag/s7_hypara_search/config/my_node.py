from enum import Enum


class SelectParaNode(Enum):
    DOE2 = "DOE2_ONE_BY_ONE"
    ExtractParaSelectorInit = "ExtractParaSelectorInit"
    FeatureSelectorInit = "FeatureSelectorInit"
    ModelParaSelectorInit = "ModelParaSelectorInit"
    EventParaSelectorInit = "EventParaSelectorInit"



"""
<1> 关于模型结果，实际统计需要展示有意义的一面。我们需要描述几个方面：整体一致性、阳性事件命中的情况lingmingdu 
<2> 要展示整体阴阳性的一致性，使用一致性的时间占比。
<3> 
"""