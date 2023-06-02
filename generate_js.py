#!/usr/bin/env python3

import argparse
import sys

JS_LOOP = r"""for (var i = 0; i < student_list.length; i++) {
  var name = student_list[i];
  var name_link = $("a").filter(function() {
    return $(this).text().startsWith(name);})[0];
  if (name_link == null) {
    console.log(name + " was not found.");
    continue;
  }
  var checkbox = name_link.parentNode.parentNode.childNodes[9].firstChild
  if (checkbox == null) {
    console.log(name + " has no checkbox.");
    continue;
  }
  $(checkbox.firstChild).prop("checked", true);
}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', nargs="?", default=sys.stdin, type=argparse.FileType("r"))
    args = parser.parse_args()

    names = [line.strip() for line in args.input]

    formatted_list = ", ".join([
        f'"{name}"' for name in names])

    print(f"var student_list = [{formatted_list}];")
    print(JS_LOOP)


if __name__ == "__main__":
    main()
