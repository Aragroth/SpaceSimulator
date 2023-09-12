import math
import time

import krpc


def main():
    conn = krpc.connect()
    execute_btn(conn)


def execute_next_node(conn):
    space_center = conn.space_center
    vessel = space_center.active_vessel
    ap = vessel.auto_pilot

    try:
        node = vessel.control.nodes[0]
    except Exception:
        return

    rf = vessel.orbit.body.reference_frame

    ap.sas = True
    time.sleep(.1)
    ap.sas_mode = vessel.auto_pilot.sas_mode.maneuver
    ap.wait()

    m = vessel.mass
    isp = vessel.specific_impulse
    dv = node.delta_v
    F = vessel.available_thrust
    G = 9.81
    burn_time = (m - (m / math.exp(dv / (isp * G)))) / (F / (isp * G))

    # Warp until burn
    space_center.warp_to(node.ut - (burn_time / 2.0) - 5.0)
    while node.time_to > (burn_time / 2.0):
        pass
    ap.wait()

    vessel.control.throttle = thrust_controller(vessel, node.remaining_delta_v)
    while node.remaining_delta_v > .1:
        ap.target_direction = node.remaining_burn_vector(rf)  # comment out this line
        vessel.control.throttle = thrust_controller(vessel, node.remaining_delta_v)

    ap.disengage()
    vessel.control.throttle = 0.0
    node.remove()


def execute_all_nodes(conn):
    space_center = conn.space_center
    vessel = space_center.active_vessel
    while vessel.control.nodes:
        execute_next_node(conn)


def thrust_controller(vessel, deltaV):
    TWR = vessel.max_thrust / vessel.mass
    if deltaV < TWR / 3:
        return .05
    elif deltaV < TWR / 2:
        return .1
    elif deltaV < TWR:
        return .25
    else:
        return 1.0


def execute_btn(conn):
    space_center = conn.space_center
    canvas = conn.ui.stock_canvas  # draw on the main screen
    panel = canvas.add_panel()  # container for our button
    rect = panel.rect_transform  # rect to define panel
    rect.size = (100, 30)  # panel size
    rect.position = (110 - (canvas.rect_transform.size[0] / 2), 0)  # left middle
    button = panel.add_button("Execute Node")  # add the button
    button.rect_transform.position = (0, 20)  # locate the button
    button_clicked = conn.add_stream(getattr, button, 'clicked')  # watch button
    while True:  # if button clicked, execute the next node
        if button_clicked():
            execute_next_node(conn)
            button.clicked = False
