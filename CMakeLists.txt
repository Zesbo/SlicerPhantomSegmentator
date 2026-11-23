#-----------------------------------------------------------------------------
set(MODULE_NAME Phantom_Segmentator)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Modul_Phantom_Segmentation.png
  Resources/Geometry/PHANTOM_GEOMETRY_BDC-6H200.json
  Resources/Geometry/PHANTOM_GEOMETRY_BDC-6H200_FOVCUT.json
  Resources/Geometry/PHANTOM_GEOMETRY_uCT_4r_V2.json
  Resources/Geometry/PHANTOM_GEOMETRY_3Dprinted_6H100_V1.json
  Resources/Geometry/PHANTOM_GEOMETRY_uCT_6H_V1.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
