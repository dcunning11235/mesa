from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from .agents import SsAgent
from .model import SugarscapeCg

from mesa.new_space import _PositionalPatchSpace

color_dic = {4: "#005C00",
             3: "#008300",
             2: "#00AA00",
             1: "#00F800",
             0: "#D6F5D6"}


def SsAgent_portrayal(obj):
    if obj is None:
        return

    portrayal = {}

    if type(obj[1]) is SsAgent:
        portrayal["Shape"] = "sugarscape_cg/resources/ant.png"
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1
    elif obj[0] == "sugar":
        portrayal["Color"] = color_dic[int(obj[1])]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal


canvas_element = CanvasGrid(SsAgent_portrayal, 50, 50, 500, 500)
chart_element = ChartModule([{"Label": "SsAgent", "Color": "#AA0000"}])

server = ModularServer(SugarscapeCg, [canvas_element, chart_element],
                       "Sugarscape 2 Constant Growback")
# server.launch()
