import numpy as np
import bokeh.io as bkio
import bokeh.plotting as bkplot
import bokeh.models as bkmodels
import bokeh.palettes as bkpalettes
from bokeh.layouts import row, column, widgetbox


# import matplotlib.pyplot as plt


def plot1image(cam, im, p0):
    ps = p0.squeeze()
    h, w = im.shape

    # Bokeh Plotting
    bkio.reset_output()
    bkio.output_file('bokeh plots.html', title='image plots')

    p = bkplot.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                      title=cam['filename'],
                      x_axis_label='pixel (1 - ' + str(w) + ')',
                      y_axis_label='pixel (1 - ' + str(h) + ')',
                      tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                      active_scroll='wheel_zoom',
                      active_inspect=None)

    # Setup tools
    hover = bkmodels.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '($x{(1.11)}, $y{(1.11)})')],
                               point_policy='follow_mouse')
    p.add_tools(hover)

    # Setup widgets
    # slider = bkmodels.widgets.Slider(start=1, end=9, value=1, step=1, title="Image")

    # Set up callbacks
    # def update_title(attrname, old, new):
    #     p.title.text = 'changed'  # str(slider.value)
    #     print('ran')
    # slider.on_change('value', update_title)

    # Plot image
    p.image(image=[np.flip(im, 0)], x=0, y=h, dw=w, dh=h, palette=bkpalettes.Greys256)

    # Plot clear rectangle
    p.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

    # Plot current KLT points
    p.circle(ps[:, 0], ps[:, 1], color='blue', legend='p0')
    p.legend.click_policy = 'hide'

    # Show plot
    # widgets = widgetbox(slider)
    # bkio.show(column(p, widgets))  # open a browser

    p2 = bkplot.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                       title=cam['filename'],
                       x_axis_label='pixel (1 - ' + str(w) + ')',
                       y_axis_label='pixel (1 - ' + str(h) + ')',
                       tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                       active_scroll='wheel_zoom',
                       active_inspect=None)
    bkio.show(p)  # open a browser
