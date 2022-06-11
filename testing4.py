
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Setup a plot such that only the bottom spine is shown
def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.tick_params(which='major', width=1.00)
    # ax.tick_params(which='major', length=5)
    # ax.tick_params(which='minor', width=0.75)
    # ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(120, 240)
    ax.set_ylim(0, 2)
    
    ax.patch.set_alpha(0.0)

plt.figure(figsize=(10, 6))
n = 6

y = [0.125, 0.125, 0.125, 0.125, 0.125]
labels = ["R10", "R20", "R50", "R100", "R200"]

# Multiple Locator
ax = plt.subplot(n, 1, 1)
setup(ax)
ax.text(0.0, 0.2, "Red Average", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [146.5186, 194.4272, 204.7196, 153.9144, 214.0696]
# Scatter plot with x and y
plt.scatter(x, y, color='r')

for i in range(5):
    ax.annotate(labels[i], (x[i], y[i]))

ax = plt.subplot(n, 1, 2)
setup(ax)
ax.text(0.0, 0.2, "Green Average", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [150.7701, 127.9269, 146.6804, 184.5635, 132.7627]

# Scatter plot with x and y
plt.scatter(x, y, color='g')

for i in range(5):
    ax.annotate(labels[i], (x[i], y[i]))

ax = plt.subplot(n, 1, 3)
setup(ax)
ax.text(0.0, 0.2, "Blue Average", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [182.3846, 163.3387, 151.6955, 166.3903, 163.8948]

# Scatter plot with x and y
plt.scatter(x, y, color='b')

ax.annotate(labels[0], (x[0], y[0]))
ax.annotate(labels[1], (x[1]+0.25, y[1]+0.25))
ax.annotate(labels[2], (x[2], y[2]))
ax.annotate(labels[3], (x[3], y[3]))
ax.annotate(labels[4], (x[4]-5, y[4]))

ax = plt.subplot(n, 1, 4)
setup(ax)
ax.text(0.0, 0.2, "Red Mode", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [197.7273, 222.9091, 231.2727, 208.5455, 230.6364]

# Scatter plot with x and y
plt.scatter(x, y, color='r')

i = 0; ax.annotate(labels[i], (x[i], y[i]))
i = 1; ax.annotate(labels[i], (x[i], y[i]))
i = 2; ax.annotate(labels[i], (x[i]+2, y[i]))
i = 3; ax.annotate(labels[i], (x[i], y[i]))
i = 4; ax.annotate(labels[i], (x[i]-2, y[i]+0.125))

ax = plt.subplot(n, 1, 5)
setup(ax)
ax.text(0.0, 0.2, "Green Mode", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [221.2728, 202.7273, 213.3636, 214, 208]

# Scatter plot with x and y
plt.scatter(x, y, color='g')

i = 0; ax.annotate(labels[i], (x[i], y[i]))
i = 1; ax.annotate(labels[i], (x[i], y[i]))
i = 2; ax.annotate(labels[i], (x[i]-1, y[i]+0.125))
i = 3; ax.annotate(labels[i], (x[i]+2, y[i]+0.125))
i = 4; ax.annotate(labels[i], (x[i], y[i]))

ax = plt.subplot(n, 1, 6)
setup(ax)
ax.text(0.0, 0.2, "Blue Mode", fontsize=14,
        transform=ax.transAxes)
# List of data points - Red
x = [189.5455, 176.4545, 188.3636, 219.7273, 179.8182]

# Scatter plot with x and y
plt.scatter(x, y, color='b')

i = 0; ax.annotate(labels[i], (x[i]-4, y[i]+0.125))
i = 1; ax.annotate(labels[i], (x[i]-4, y[i]+0.125))
i = 2; ax.annotate(labels[i], (x[i]+3, y[i]))
i = 3; ax.annotate(labels[i], (x[i], y[i]))
i = 4; ax.annotate(labels[i], (x[i], y[i]))


# Push the top of the top axes outside the figure because we only show the
# bottom spine.
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=1.05)

plt.show()