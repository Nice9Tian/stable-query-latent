// pybind11 first: Python headers must not see Qt's `slots` macro.
#include <pybind11/embed.h>

#include "mainwindow.h"

#include <QApplication>
#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

namespace py = pybind11;

// pybind11's scoped_interpreter holds the GIL for as long as it lives; worker
// threads acquire it per call via py::gil_scoped_acquire. Both objects are
// intentionally leaked: tearing the interpreter down under a live QThread at
// exit is a crash waiting to happen (pattern from CodeX2Thirdpart).
static py::scoped_interpreter *gInterpreter = nullptr;
static py::gil_scoped_release *gReleaseMainGil = nullptr;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Bundled Python: python312/ beside the exe wins; the source-tree copy
    // (tools/python312) is the Qt-Creator-run fallback.
    const QDir appDir(QCoreApplication::applicationDirPath());
    QString pythonRoot = appDir.absoluteFilePath("python312");
    if (!QFileInfo::exists(pythonRoot + "/python.exe"))
        pythonRoot = QStringLiteral(LARICE_LOCAL_PYTHON_ROOT);
    if (!QFileInfo::exists(pythonRoot + "/python.exe")) {
        QMessageBox::critical(nullptr, QStringLiteral("Larice"),
            QStringLiteral("未找到内置 Python（python312/）。\n"
                           "请先运行 windows/setup_python_pybind11.bat。"));
        return 1;
    }
    qputenv("PYTHONHOME", pythonRoot.toUtf8());
    qputenv("PYTHONPATH", QDir(pythonRoot).absoluteFilePath("Lib").toUtf8());

    gInterpreter = new py::scoped_interpreter{};
    {
        // larice_bridge.py: beside the exe first, source tree as fallback.
        auto sysPath = py::module_::import("sys").attr("path");
        sysPath.attr("insert")(0, appDir.absolutePath().toStdString());
        sysPath.attr("insert")(0, std::string(LARICE_SOURCE_DIR));
    }
    gReleaseMainGil = new py::gil_scoped_release{};

    MainWindow w;
    w.show();
    return a.exec();
}
