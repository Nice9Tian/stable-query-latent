# Copy the bundled Python's Lib/ into the dist package WITHOUT dev-installed
# site-packages (torch alone is ~3 GB). Runtime deps are pip-installed on the
# target machine at first launch; only pip itself ships.
#   cmake -DSRC=<python312/Lib> -DDST=<dist .../python312/Lib> -P copy_python_lib.cmake

if(NOT SRC OR NOT DST)
    message(FATAL_ERROR "copy_python_lib.cmake needs -DSRC and -DDST")
endif()

file(REMOVE_RECURSE "${DST}")
file(COPY "${SRC}/" DESTINATION "${DST}" PATTERN "site-packages" EXCLUDE)

file(GLOB PIP_ITEMS "${SRC}/site-packages/pip" "${SRC}/site-packages/pip-*"
     "${SRC}/site-packages/README.txt")
file(MAKE_DIRECTORY "${DST}/site-packages")
if(PIP_ITEMS)
    file(COPY ${PIP_ITEMS} DESTINATION "${DST}/site-packages")
endif()
