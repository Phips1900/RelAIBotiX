import importlib
import inspect
import xml.etree.ElementTree as ET
import os
from models import __path__ as models_path

ROBOT_CLASSES = {}

class Model:
    def __init__(self, path):
        self.scene_path = path
        self.path = None


    def set_model_xml(self):
        """
        Set the mujoco-xml file from '/scene.xml' to the corresponding model file
        :return: Output the model file from the corresponding model directory
        """
        tree = ET.parse(self.scene_path)
        root = tree.getroot()

        model_name = root.find(".//include").get("file")
        xml_dir = os.path.dirname(self.scene_path)
        model_file = os.path.join(xml_dir, model_name)

        self.path = model_file
        return model_file


    def get_model_name(self):
        """
        Read the model name from the mujoco-xml file
        :param xml_file: Input the mjcf file path
        :return: Output the model name
        """
        path = self.set_model_xml()
        tree = ET.parse(path)
        root = tree.getroot()

        name = root.findall(".//default/*")[0].get("class")
        # name = root.get("model")
        return name


    def get_dof(self):
        """
        Parse the mujoco-xml file for a robotic model and return the degrees of freedom (DoF)
        :param path: Input the mjcf file path
        :param ignore_keywords: Input the attribute names of joints not associated with the kinematics.py
        :return: Output the degrees of freedom
        """
        ignore_keywords = ["jaw", "finger", "pad", "gripper", "follower", "driver", "spring", "coupler"]

        path = self.set_model_xml()
        tree = ET.parse(path)
        root = tree.getroot()

        actuated_joints = set()
        actuated_tendons = set()

        # Collect model name
        name = root.findall(".//default/*")[0].get("class")
        print("Model Name: ", name)

        # Collect actuator and tendons
        for actuator in root.findall(".//actuator/*"):
            joint = actuator.get("joint")
            if joint:
                actuated_joints.add(joint)
            tendon = actuator.get("tendon")
            if tendon:
                actuated_tendons.add(tendon)

        # For the finger joints in panda
        for tendon in root.findall(".//tendon/fixed"):
            tendon_name = tendon.get("name")
            if tendon_name in actuated_tendons:
                for j in tendon.findall("joint"):
                    joint_name = j.get("joint")
                    if joint_name:
                        actuated_joints.add(joint_name)

        dof_count = 0
        joint_details = []

        for joint in root.findall(".//joint"):
            name = joint.get("name")
            joint_type = joint.get("type", "hinge")

            # Ignore joints not in actuator
            if name not in actuated_joints:
                continue

            # Filter out gripper/finger joints
            if any(keyword.lower() in name.lower() for keyword in ignore_keywords):
                continue

            if joint_type in ["hinge", "slide"]:
                dof_count += 1
                joint_details.append((name, joint_type, 1))
            elif joint_type == "ball":
                dof_count += 3
                joint_details.append((name, joint_type, 3))
            elif joint_type == "free":
                dof_count += 6
                joint_details.append((name, joint_type, 6))

        print(f"Actuated joints found: {sorted(actuated_joints)}")
        print("Counted joints and DOF:")
        for jname, jtype, jval in joint_details:
            print(f"{jname} ({jtype}): {jval} DOF")
        print(f"\nDegrees of Freedom: {dof_count}")
        return dof_count


    def get_model(self):
        """
        Reads the models package and returns the class of the required robot model
        :return: Output the class associated with the desired robot model
        """
        model_name = self.get_model_name()

        for filename in os.listdir(models_path[0]):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"models.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, cls in inspect.getmembers(module, inspect.isclass):
                        if cls.__module__ == module_name:
                            ROBOT_CLASSES[filename[:-3]] = cls
                            break
                except Exception as e:
                    print(f"Warning: failed to import '{module_name}': {e}")

        robot = ROBOT_CLASSES.get(model_name)
        return robot()


    def get_model_list(self):
        """
        Reads the models package and returns the class of the required robot model
        :return: Output the class associated with the desired robot model
        """
        # model_name = self.get_model_name()

        for filename in os.listdir(models_path[0]):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"models.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, cls in inspect.getmembers(module, inspect.isclass):
                        if cls.__module__ == module_name:
                            ROBOT_CLASSES[filename[:-3]] = cls
                            break
                except Exception as e:
                    print(f"Warning: failed to import '{module_name}': {e}")

        # robot = ROBOT_CLASSES.get(model_name)
        return ROBOT_CLASSES

    def get_path_from_models_package(self, robot_name: str):
        """
        Get the XML path by instantiating the model class corresponding to robot_name
        :param robot_name: Name of the robot as specified in the xml
        :return:
        """
        # Ensure if the ROBOT_CLASSES are loaded
        if not ROBOT_CLASSES:
            self.get_model_list()

        robot_cls = ROBOT_CLASSES.get(robot_name.lower())
        if not robot_cls:
            raise ValueError(f"No robot found with name '{robot_name}' in the package")

        robot_instance = robot_cls()
        return robot_instance.xml_path


    def parse_xml(self, body_elem, parent_name, urdf_links, urdf_joints):
        """
        Deprecated
        Parse a MuJoCo XML element and convert it into URDF 'link' and 'joint' element
        :param body_elem: The XML element representing a MuJoCo '<body>'
        :param parent_name: The name of the parent link to which the current link is connected
        :param urdf_links: A list to which the generated URDF '<link>' element will be appended.
        :param urdf_joints: A list to which the generated URDF '<joint>' elements will be appended
        """
        link_name = body_elem.get("name", "unnamed").replace(" ", "_")

        link = ET.Element("link", name=link_name)
        urdf_links.append(link)

        for joint_elem in body_elem.findall("joint"):
            joint_name = joint_elem.get("name", f"joint_{link_name}")
            joint_type = joint_elem.get("type", "revolute")
            joint = ET.Element("joint", name=joint_name, type=joint_type)

            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=link_name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 0 1")

            range_attr = joint_elem.get("range")
            if range_attr:
                try:
                    lower, upper = map(float, range_attr.strip().split())
                    ET.SubElement(joint, "limit", lower=str(lower), upper=str(upper), effort="10.0", velocity="1.0")
                except Exception as e:
                    print(f"Warning: Could not parse range for joint '{joint_name}': {range_attr} — {e}")
                    ET.SubElement(joint, "limit", lower="-3.14", upper="3.14", effort="10.0", velocity="1.0")
            else:
                ET.SubElement(joint, "limit", lower="-3.14", upper="3.14", effort="10.0", velocity="1.0")

            urdf_joints.append(joint)
            urdf_joints.append(joint)

        for child in body_elem.findall("body"):
            self.parse_xml(child, link_name, urdf_links, urdf_joints)

    def mujoco_to_urdf(self, input_xml_path, output_urdf_path):
        """
        Deprecated
        Convert MuJoCo XML file to the corresponding URDF file to obtain the ETS sequence for the kinematic chain.
        :param input_xml_path: Input the path of the MuJoCo XML model file which needs to be converted
        :param output_urdf_path: Input the path of the URDF file to store the converted file.
        """
        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No <worldbody> found in the Mujoco XML")

        base_body = worldbody.find("body")
        if base_body is None:
            raise ValueError("No base <body> found in the Mujoco XML")

        robot = ET.Element("robot", name="converted_robot")
        ET.SubElement(robot, "link", name="world")

        urdf_links = []
        urdf_joints = []

        self.parse_xml(base_body, "world", urdf_links, urdf_joints)

        for link in urdf_links:
            robot.append(link)
        for joint in urdf_joints:
            robot.append(joint)

        urdf_str = ET.tostring(robot, encoding="unicode")
        with open(output_urdf_path, "w") as f:
            f.write(urdf_str)

        print(f"URDF written to: {output_urdf_path}")



