if(NOT DEPLOY_CONFIG MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
    return()
endif()

execute_process(
    COMMAND "${WINDEPLOYQT_EXECUTABLE}" --release "${TARGET_FILE}"
    RESULT_VARIABLE deploy_result
)

if(NOT deploy_result EQUAL 0)
    message(FATAL_ERROR "windeployqt failed for ${TARGET_FILE}")
endif()
