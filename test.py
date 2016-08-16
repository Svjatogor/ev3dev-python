#!/usr/bin/env python

# Hard coding these values is not a good idea because the values could
# change. But, since this is an example, we want to keep it short.

import os
import array

from number_for_lcd import NumberDrawer

def main():
	num = NumberDrawer()
	num.draw_number(9)

if __name__ == '__main__':
    main()
