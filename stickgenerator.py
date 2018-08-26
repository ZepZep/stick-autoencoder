from PIL import Image, ImageDraw
import math

from imageshower import show


def get_stick(parameters, dim=(128, 128)):
    image = Image.new("1", dim)
    draw = ImageDraw.Draw(image)

    par = {
        "head_angle":       parameters[0],
        "hand_l_angle":     parameters[1],
        "hand_r_angle":     parameters[2],
        "leg_l_angle":      parameters[3],
        "leg_r_angle":      parameters[4],
    }


    cx = 64

    head_top = 44
    head_r = 18

    body_top = 40
    body_dim = (8, 40)

    hand_top = 46
    hand_dim = (6, 40)
    hand_angle = math.pi / 6

    leg_top = 80
    leg_dim = (6, 40)
    leg_angle = math.pi/12

    draw.rectangle((0 , 0, dim[0]-1, dim[1]-1), outline=1)

    _draw_circle_bc(draw, (cx, head_top), head_r, par["head_angle"])
    _draw_rect_tc(draw, (cx, body_top), body_dim, 0)

    # hands
    _draw_rect_tc(draw, (cx, hand_top), hand_dim, hand_angle + par["hand_l_angle"])
    _draw_rect_tc(draw, (cx, hand_top), hand_dim, -hand_angle + par["hand_r_angle"])

    # legs
    _draw_rect_tc(draw, (cx, leg_top), leg_dim, leg_angle + par["leg_l_angle"])
    _draw_rect_tc(draw, (cx, leg_top), leg_dim, -leg_angle + par["leg_r_angle"])

    return image


def _draw_circle_bc(draw, bc, r, angle):
    x, y = _rotate_points([(0, -r)], angle, bc)[0]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=1, outline=1)

def _draw_rect_tc(draw, tc, dim, angle):
    w, h = dim
    points = [(-w/2, 0), (-w/2, h), (w/2, h), (w/2, 0)]
    points = _rotate_points(points, angle, tc)
    draw.polygon(points, fill=1, outline=1)

def _rotate_points(points, angle, offset=(0, 0)):
    c, s = math.cos(angle), math.sin(angle)
    return [(c * x - s * y + offset[0], s * x + c * y + offset[1]) for (x, y) in points]