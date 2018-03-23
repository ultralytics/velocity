import numpy as np
import bokeh.io as bkio
import bokeh.plotting as bkplot
import bokeh.models as bkmodels
import bokeh.palettes as bkpalettes
from bokeh.layouts import row, column, widgetbox


# import matplotlib.pyplot as plt


def plot1image(cam, im, P):
    # Bokeh Plotting
    bkio.reset_output()
    bkio.output_file('bokeh plots.html', title='image plots')
    h, w = im.shape
    n = P.shape[2]
    colors = bokeh_colors(n)

    x = P[0,]  # x
    y = P[1,]  # y
    v = P[2,]  # valid

    p = bkplot.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(900 * w / h), plot_height=900,
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

    # Plot license plate outline
    p.patch(x[0:4, 0], y[0:4, 0], alpha=0.3, line_width=4)

    # Plot points
    # p.circle(P[0,].ravel(), P[1,].ravel(), color='blue', legend='KLT Points')
    for i in np.arange(P.shape[2]):
        p.circle(x[:, i], y[:, i], color=colors[i], legend='image ' + str(i))

    # Plot lines
    p.multi_line(x.tolist(), y.tolist(), color='white', alpha=.8, line_width=1)

    # Show plot
    # widgets = widgetbox(slider)
    # bkio.show(column(p, widgets))  # open a browser
    p.legend.click_policy = 'hide'
    bkio.show(p)  # open a browser


def bokeh_colors(n):
    # https: // bokeh.pydata.org / en / latest / docs / reference / palettes.html
    # returns appropriate 10, 20 or 256 colors for plotting. n is the maximum required colors
    if n < 11:
        return bkpalettes.Category10[10]
    elif n < 21:
        return bkpalettes.Category20[20]
    else:
        return bkpalettes.Viridis256
