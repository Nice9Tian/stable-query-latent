#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QJsonObject>
#include <QMainWindow>
#include <QThread>
#include <QVector>

QT_BEGIN_NAMESPACE
class QPlainTextEdit;
class QPushButton;
class QTableWidget;
class QLabel;
class QMenu;
class QAction;
QT_END_NAMESPACE

// All Python work happens on this object's thread: pip bootstrap, model load,
// embedding + prediction, CSV export. Each slot grabs the GIL for its call.
class PyBridge : public QObject
{
    Q_OBJECT

public slots:
    void initPython();
    void bootstrapAndLoad(const QString &assetsDir);
    void embedText(const QString &text);
    void exportData(const QString &rowsJson, const QString &path,
                    const QString &format);

signals:
    void progress(const QString &line);
    void modelReady(const QString &metaJson);
    void embedFinished(const QString &resultJson);
    void exportFinished(const QString &resultJson);
    void failed(const QString &error);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

signals:
    void requestBootstrap(const QString &assetsDir);
    void requestEmbed(const QString &text);
    void requestExport(const QString &rowsJson, const QString &path,
                       const QString &format);

private slots:
    void onProgress(const QString &line);
    void onModelReady(const QString &metaJson);
    void onEmbedFinished(const QString &resultJson);
    void onExportFinished(const QString &resultJson);
    void onFailed(const QString &error);

private:
    enum class Lang { En = 0, Zh = 1, Ja = 2 };

    struct AnchorEntry {
        QVector<double> anchor;
        QString gamesJson;   // predictions serialized for re-display on click
        QString tagsJson;
    };

    void buildUi();
    void buildMenu();
    void setLanguage(Lang lang);
    void retranslate();
    void showReadyStatus();
    void doEmbed();
    void doExport();
    void anchorContextMenu(const QPoint &pos);
    void deleteAnchorRow(int row);
    void showPredictions(const QString &gamesJson, const QString &tagsJson);
    QString anchorRowsJson() const;
    QString assetsDir() const;

    QPlainTextEdit *descEdit = nullptr;
    QPushButton *embedBtn = nullptr;
    QTableWidget *anchorTable = nullptr;
    QPushButton *exportBtn = nullptr;
    QTableWidget *gameTable = nullptr;
    QTableWidget *tagTable = nullptr;
    QLabel *statusLabel = nullptr;
    QLabel *descLabel = nullptr;
    QLabel *anchorsLabel = nullptr;
    QLabel *gamesLabel = nullptr;
    QLabel *tagsLabel = nullptr;
    QMenu *langMenu = nullptr;
    QAction *actEn = nullptr;
    QAction *actZh = nullptr;
    QAction *actJa = nullptr;

    QThread workerThread;
    PyBridge *bridge = nullptr;
    QVector<AnchorEntry> entries;   // parallel to anchorTable rows
    bool modelLoaded = false;
    Lang lang = Lang::En;
    QJsonObject readyMeta;          // last modelReady payload (for retranslate)
};

#endif // MAINWINDOW_H
