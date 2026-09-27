"""Feeding environment —— 机器人辅助喂食仿真环境.

基于 RCareWorld 的 bathing_env.py 结构改写。
场景来自你打包的 FeedingPlayer.x86_64(源自 RCareCommon/Example Scenes/Feeding.unity)。

场景内容:
  - kinova_gen3_with_spoon : 带勺子的 Kinova 机械臂
  - food                   : 食物
  - SMPLX_male             : 坐在轮椅上的人(SMPL-X 人体,中风病人)
  - WheelChair             : 轮椅
  - Mouth                  : 嘴(喂食目标点,挂在人体 jaw 骨骼下,手动添加的 GameObjectAttr)

场景物体 ID(在 Unity Inspector 的 Attr 组件里查到):
  机械臂 kinova_gen3_with_spoon : 315893  (ControllerAttr)
  食物   food                   : 19024   (RigidbodyAttr)
  人体   SMPLX_male             : 2333    (HumanArticulationAttr)
  嘴     Mouth                  : 9999    (GameObjectAttr, 手动添加)

用法:
  from feeding_env import FeedingEnv
  env = FeedingEnv(graphics=True)      # 带画面
  env = FeedingEnv(graphics=False)     # 无画面(训练时更快)
"""

from pyrcareworld.attributes.controller_attr import ControllerAttr
from pyrcareworld.envs.base_env import RCareWorld


# 你打包的可执行文件路径(如果换了机器/路径,改这里)
_DEFAULT_EXECUTABLE_PATH = (
    "/home/sophia/research/my_feeding_project/FeedingBuild/FeedingPlayer_v2.x86_64"
)
# Build before 2026-09-25 (no working spoon collider, invisible body 'man' enabled).
_V1_EXECUTABLE_PATH = (
    "/home/sophia/research/my_feeding_project/FeedingBuild/FeedingPlayer.x86_64"
)


class FeedingEnv(RCareWorld):
    """机器人辅助喂食环境。

    人坐在轮椅上(固定,不动),机械臂拿勺子把食物送到嘴边,
    同时接触力需保持安全(参考擦浴阈值 1-6N)。
    """

    # ---- Unity 物体 ID ----
    _robot_id: int = 315893   # kinova_gen3_with_spoon(机械臂)
    _food_id: int = 19024     # food(食物)
    _person_id: int = 2333    # SMPLX_male(人体, HumanArticulationAttr)
    _mouth_id: int = 9999     # Mouth(喂食目标点)
    _spoon_id: int = 114514   # spoon (RigidbodyAttr, kinematic in the scene)
    _spoon_link_index: int = 8  # robot link the spoon is attached to (EndEffector_Link)

    def __init__(
        self,
        executable_file: str = _DEFAULT_EXECUTABLE_PATH,
        seed: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(executable_file=executable_file, *args, **kwargs)

    # ---- 访问器:根据 ID 拿到场景里的物体 ----

    def get_robot(self) -> ControllerAttr:
        """拿到带勺子的机械臂。"""
        return self.GetAttr(self._robot_id)

    def get_food(self):
        """拿到食物对象。用 .data['position'] 读它的位置。"""
        return self.GetAttr(self._food_id)

    def get_mouth(self):
        """拿到嘴(喂食目标点)。用 .data['position'] 读目标坐标。"""
        return self.GetAttr(self._mouth_id)

    def attach_spoon(self):
        """Make the spoon a dynamic body fixed to the end effector, so it collides.

        The scene keeps the spoon kinematic, and kinematic bodies pass through
        the face colliders. Order matters (measured 2026-09-25): Link while still
        kinematic, then make it dynamic; the other order moves the spoon ~41 cm
        away. Afterwards it follows the arm to < 1 mm.
        """
        spoon = self.GetAttr(self._spoon_id)
        spoon.Link(self._robot_id, self._spoon_link_index)
        self.step(2)
        spoon.SetUseGravity(False)
        spoon.SetMass(0.05)
        spoon.SetKinematic(False)
        self.step(10)
        return spoon

    def get_person(self):
        """拿到人体(SMPL-X)。可读各骨骼位置(head/jaw 等)。"""
        return self.GetAttr(self._person_id)

    # ---- 便捷方法 ----

    def get_mouth_position(self):
        """返回嘴的位置 [x, y, z](喂食目标点)。"""
        self.step()
        return self.get_mouth().data.get("position")

    def get_robot_position(self):
        """返回机械臂末端/根位置 [x, y, z]。"""
        self.step()
        return self.get_robot().data.get("position")

    def get_food_position(self):
        """返回食物位置 [x, y, z]。"""
        self.step()
        return self.get_food().data.get("position")


# ---- 最小连接测试:确认能连上场景并读到各物体位置 ----
if __name__ == "__main__":
    print("正在启动喂食环境...")
    env = FeedingEnv(graphics=True)   # 想无画面测试就改成 graphics=False
    env.step()

    print("\n========== 读取场景物体 ==========")
    robot = env.get_robot()
    food = env.get_food()
    mouth = env.get_mouth()
    person = env.get_person()
    env.step()

    print("机械臂 (315893) position:", robot.data.get("position"))
    print("食物   (19024)  position:", food.data.get("position"))
    print("嘴     (9999)   position:", mouth.data.get("position"))
    print("人体   (2333)   position:", person.data.get("position"))

    print("\n如果上面都打印出了坐标(不是 None),说明你的喂食环境连接成功!")

    env.close()
