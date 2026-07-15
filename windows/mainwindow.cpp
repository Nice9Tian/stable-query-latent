// pybind11 first: Python headers must not see Qt's `slots` macro.
#include <pybind11/embed.h>

#include "mainwindow.h"

#include <QActionGroup>
#include <QClipboard>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>

namespace py = pybind11;

// ------------------------------ UI strings ---------------------------------
// Three built-in languages; the app defaults to English, the menu switches.

namespace {

enum Str {
    SWinTitle, SDescLabel, SDescPlaceholder, SEmbedBtn, SAnchorsLabel,
    SColName, SColAnchor, SExportBtn, SGamesLabel, SColGame, SColScore,
    STagsLabel, SColTag, SColLikelihood, SStarting, SEmbedding,
    SReadyFmt, SDoneFmt, SExportedFmt, SErrorStatus, SErrorTitle,
    SExportDialogTitle, SCsvFilter, SAnchorDefaultFmt, SLangMenu,
    SCount
};

const char *T[3][SCount] = {
    {   // English
        "Larice Anchor Studio — champion tower 2ice_cegate",
        "Game description (multi-line)",
        "Type a game description here, then press the button below to get its embedding anchor…",
        "Embed → Anchor",
        "Session anchors (click a name to rename, click a row to review)",
        "Name", "Anchor",
        "Export… (CSV / JSON)",
        "Predicted games (by score, descending)",
        "Game", "Score",
        "Predicted tags (by likelihood, descending)",
        "Tag", "Likelihood",
        "Starting…",
        "Embedding…",
        "Ready | device %1 | %2 game anchors | %3 tags | all-game anchor IDs saved to %4",
        "Done: %1 sentence(s) → anchor %2",
        "Exported %1 row(s) → %2",
        "Error (see dialog)",
        "Larice",
        "Export anchors CSV",
        "CSV files (*.csv)",
        "Anchor %1",
        "&Language",
    },
    {   // 中文
        "Larice 锚点工作台 — 冠军塔 2ice_cegate",
        "游戏描述（可多行）",
        "在此输入一段游戏描述，点击下方按钮得到它的嵌入锚点……",
        "嵌入 → 锚点",
        "本次会话的锚点（点击名字可改，点击行回看预测）",
        "名字", "锚点",
        "批量导出…（CSV / JSON）",
        "预测游戏（按得分从大到小）",
        "游戏", "得分",
        "预测标签（按可能性从大到小）",
        "标签", "可能性",
        "启动中……",
        "嵌入中……",
        "就绪 | 设备 %1 | %2 个游戏锚 | %3 个标签 | 全游戏锚 ID 已存至 %4",
        "完成：%1 句 → 锚点 %2",
        "已导出 %1 行 → %2",
        "出错（详情见弹窗）",
        "Larice",
        "导出锚点 CSV",
        "CSV 文件 (*.csv)",
        "锚点 %1",
        "语言(&L)",
    },
    {   // 日本語
        "Larice アンカースタジオ — チャンピオンタワー 2ice_cegate",
        "ゲーム説明（複数行可）",
        "ここにゲームの説明を入力し、下のボタンで埋め込みアンカーを取得します…",
        "埋め込み → アンカー",
        "セッションのアンカー（名前をクリックで変更、行クリックで予測を再表示）",
        "名前", "アンカー",
        "エクスポート…（CSV / JSON）",
        "予測ゲーム（スコア降順）",
        "ゲーム", "スコア",
        "予測タグ（可能性降順）",
        "タグ", "可能性",
        "起動中…",
        "埋め込み中…",
        "準備完了 | デバイス %1 | ゲームアンカー %2 件 | タグ %3 件 | 全ゲームのアンカー ID を %4 に保存",
        "完了：%1 文 → アンカー %2",
        "%1 行をエクスポート → %2",
        "エラー（ダイアログ参照）",
        "Larice",
        "アンカー CSV をエクスポート",
        "CSV ファイル (*.csv)",
        "アンカー %1",
        "言語(&L)",
    },
};

}   // namespace

#define S(id) QString::fromUtf8(T[int(lang)][id])

// ---------------------------------------------------------------------------
// PyBridge — every slot runs on the worker thread and takes the GIL itself.
// ---------------------------------------------------------------------------

void PyBridge::initPython()
{
    // Register a PERMANENT CPython thread state for this worker thread.
    // Without it, each py::gil_scoped_acquire creates and then DESTROYS a
    // thread state; torch's own embedded pybind11 caches a pointer to the
    // first one and self-deadlocks in device_lazy_init on the next call
    // (observed: PyEval_AcquireThread waiting forever under BatchEncoding.to).
    // Ensure once, release the GIL, and intentionally never PyGILState_Release
    // -- the tstate must live as long as the thread.
    PyGILState_Ensure();
    PyEval_SaveThread();
}

void PyBridge::bootstrapAndLoad(const QString &assetsDir)
{
    try {
        py::gil_scoped_acquire gil;
        auto cb = py::cpp_function([this](const std::string &line) {
            emit progress(QString::fromStdString(line));
        });
        py::module_ mod = py::module_::import("larice_bridge");

        QString r = QString::fromStdString(py::cast<std::string>(
            mod.attr("bootstrap")(cb)));
        QJsonObject o = QJsonDocument::fromJson(r.toUtf8()).object();
        if (!o.value("ok").toBool()) {
            emit failed(o.value("error").toString());
            return;
        }

        r = QString::fromStdString(py::cast<std::string>(
            mod.attr("load_model")(assetsDir.toStdString(), cb)));
        o = QJsonDocument::fromJson(r.toUtf8()).object();
        if (!o.value("ok").toBool()) {
            emit failed(o.value("error").toString());
            return;
        }
        emit modelReady(r);
    } catch (const py::error_already_set &e) {
        emit failed(QString::fromUtf8(e.what()));
    }
}

void PyBridge::embedText(const QString &text)
{
    try {
        py::gil_scoped_acquire gil;
        py::module_ mod = py::module_::import("larice_bridge");
        QString r = QString::fromStdString(py::cast<std::string>(
            mod.attr("embed_and_predict")(text.toStdString())));
        emit embedFinished(r);
    } catch (const py::error_already_set &e) {
        emit failed(QString::fromUtf8(e.what()));
    }
}

void PyBridge::exportData(const QString &rowsJson, const QString &path,
                          const QString &format)
{
    try {
        py::gil_scoped_acquire gil;
        py::module_ mod = py::module_::import("larice_bridge");
        const char *fn = (format == QStringLiteral("json"))
                             ? "export_anchors_json" : "export_anchors_csv";
        QString r = QString::fromStdString(py::cast<std::string>(
            mod.attr(fn)(rowsJson.toStdString(), path.toStdString())));
        emit exportFinished(r);
    } catch (const py::error_already_set &e) {
        emit failed(QString::fromUtf8(e.what()));
    }
}

// ---------------------------------------------------------------------------
// MainWindow
// ---------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    const int saved = QSettings(QStringLiteral("larice"),
                                QStringLiteral("anchor-studio"))
                          .value(QStringLiteral("lang"), 0).toInt();
    lang = static_cast<Lang>(qBound(0, saved, 2));

    buildUi();
    buildMenu();
    retranslate();

    bridge = new PyBridge;
    bridge->moveToThread(&workerThread);
    connect(&workerThread, &QThread::finished, bridge, &QObject::deleteLater);
    connect(this, &MainWindow::requestBootstrap, bridge, &PyBridge::bootstrapAndLoad);
    connect(this, &MainWindow::requestEmbed, bridge, &PyBridge::embedText);
    connect(this, &MainWindow::requestExport, bridge, &PyBridge::exportData);
    connect(bridge, &PyBridge::progress, this, &MainWindow::onProgress);
    connect(bridge, &PyBridge::modelReady, this, &MainWindow::onModelReady);
    connect(bridge, &PyBridge::embedFinished, this, &MainWindow::onEmbedFinished);
    connect(bridge, &PyBridge::exportFinished, this, &MainWindow::onExportFinished);
    connect(bridge, &PyBridge::failed, this, &MainWindow::onFailed);
    workerThread.start();

    // Must run FIRST on the worker thread (queued order is preserved).
    QMetaObject::invokeMethod(bridge, &PyBridge::initPython,
                              Qt::QueuedConnection);
    emit requestBootstrap(assetsDir());
}

MainWindow::~MainWindow()
{
    workerThread.quit();
    workerThread.wait(3000);
}

QString MainWindow::assetsDir() const
{
    const QString env = QProcessEnvironment::systemEnvironment()
                            .value(QStringLiteral("LARICE_ASSETS_DIR"));
    if (!env.isEmpty())
        return env;
    // dist layout: assets/ ships beside the app exe (resources/assets)
    const QString beside = QCoreApplication::applicationDirPath()
                           + QStringLiteral("/assets");
    if (QFileInfo::exists(beside + QStringLiteral("/tower.pt")))
        return beside;
    return QStringLiteral(LARICE_SOURCE_DIR) + QStringLiteral("/assets");
}

void MainWindow::buildUi()
{
    resize(1180, 800);

    // ---- top left: description input ----
    auto *inputBox = new QWidget;
    auto *inputLay = new QVBoxLayout(inputBox);
    descLabel = new QLabel;
    inputLay->addWidget(descLabel);
    descEdit = new QPlainTextEdit;
    inputLay->addWidget(descEdit, 1);
    embedBtn = new QPushButton;
    embedBtn->setEnabled(false);
    connect(embedBtn, &QPushButton::clicked, this, &MainWindow::doEmbed);
    inputLay->addWidget(embedBtn);

    // ---- top right: session anchor list + export ----
    auto *anchorBox = new QWidget;
    auto *anchorLay = new QVBoxLayout(anchorBox);
    anchorsLabel = new QLabel;
    anchorLay->addWidget(anchorsLabel);
    anchorTable = new QTableWidget(0, 2);
    anchorTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Interactive);
    anchorTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    anchorTable->setColumnWidth(0, 180);
    anchorTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    anchorTable->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(anchorTable, &QTableWidget::customContextMenuRequested,
            this, &MainWindow::anchorContextMenu);
    connect(anchorTable, &QTableWidget::currentCellChanged, this,
            [this](int row, int, int, int) {
                if (row >= 0 && row < entries.size())
                    showPredictions(entries[row].gamesJson, entries[row].tagsJson);
            });
    anchorLay->addWidget(anchorTable, 1);
    exportBtn = new QPushButton;
    exportBtn->setEnabled(false);
    connect(exportBtn, &QPushButton::clicked, this, &MainWindow::doExport);
    anchorLay->addWidget(exportBtn);

    auto *topSplit = new QSplitter(Qt::Horizontal);
    topSplit->addWidget(inputBox);
    topSplit->addWidget(anchorBox);
    topSplit->setStretchFactor(0, 1);
    topSplit->setStretchFactor(1, 1);

    // ---- bottom: predicted games / predicted tags ----
    auto makePredTable = [](QLabel *&label, QTableWidget *&table) {
        auto *box = new QWidget;
        auto *lay = new QVBoxLayout(box);
        label = new QLabel;
        lay->addWidget(label);
        table = new QTableWidget(0, 2);
        table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        table->setColumnWidth(1, 110);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionBehavior(QAbstractItemView::SelectRows);
        lay->addWidget(table, 1);
        return box;
    };
    QWidget *gameBox = makePredTable(gamesLabel, gameTable);
    QWidget *tagBox = makePredTable(tagsLabel, tagTable);

    auto *bottomSplit = new QSplitter(Qt::Horizontal);
    bottomSplit->addWidget(gameBox);
    bottomSplit->addWidget(tagBox);

    auto *mainSplit = new QSplitter(Qt::Vertical);
    mainSplit->addWidget(topSplit);
    mainSplit->addWidget(bottomSplit);
    mainSplit->setStretchFactor(0, 3);
    mainSplit->setStretchFactor(1, 2);
    setCentralWidget(mainSplit);

    statusLabel = new QLabel;
    statusBar()->addWidget(statusLabel, 1);
}

void MainWindow::buildMenu()
{
    langMenu = menuBar()->addMenu(QString());
    auto *group = new QActionGroup(this);
    group->setExclusive(true);
    auto addLang = [&](const QString &native, Lang l) {
        QAction *a = langMenu->addAction(native);
        a->setCheckable(true);
        a->setChecked(lang == l);
        group->addAction(a);
        connect(a, &QAction::triggered, this, [this, l] { setLanguage(l); });
        return a;
    };
    actEn = addLang(QStringLiteral("English"), Lang::En);
    actZh = addLang(QStringLiteral("中文"), Lang::Zh);
    actJa = addLang(QStringLiteral("日本語"), Lang::Ja);
}

void MainWindow::setLanguage(Lang l)
{
    if (lang == l)
        return;
    lang = l;
    QSettings(QStringLiteral("larice"), QStringLiteral("anchor-studio"))
        .setValue(QStringLiteral("lang"), int(l));
    retranslate();
}

void MainWindow::retranslate()
{
    setWindowTitle(S(SWinTitle));
    descLabel->setText(S(SDescLabel));
    descEdit->setPlaceholderText(S(SDescPlaceholder));
    embedBtn->setText(S(SEmbedBtn));
    anchorsLabel->setText(S(SAnchorsLabel));
    anchorTable->setHorizontalHeaderLabels({S(SColName), S(SColAnchor)});
    exportBtn->setText(S(SExportBtn));
    gamesLabel->setText(S(SGamesLabel));
    gameTable->setHorizontalHeaderLabels({S(SColGame), S(SColScore)});
    tagsLabel->setText(S(STagsLabel));
    tagTable->setHorizontalHeaderLabels({S(SColTag), S(SColLikelihood)});
    langMenu->setTitle(S(SLangMenu));
    if (modelLoaded)
        showReadyStatus();
    else if (statusLabel->text().isEmpty())
        statusLabel->setText(S(SStarting));
}

void MainWindow::showReadyStatus()
{
    statusLabel->setText(S(SReadyFmt)
                             .arg(readyMeta.value("device").toString())
                             .arg(readyMeta.value("n_games").toInt())
                             .arg(readyMeta.value("n_tags").toInt())
                             .arg(readyMeta.value("games_csv").toString()));
}

// ---------------------------------------------------------------------------

void MainWindow::onProgress(const QString &line)
{
    statusLabel->setText(line);
}

void MainWindow::onModelReady(const QString &metaJson)
{
    readyMeta = QJsonDocument::fromJson(metaJson.toUtf8()).object();
    modelLoaded = true;
    embedBtn->setEnabled(true);
    showReadyStatus();
    if (qEnvironmentVariableIsSet("LARICE_AUTOTEST")) {   // headless smoke hook
        descEdit->setPlainText(QStringLiteral(
            "A cooperative horror game about exploring abandoned facilities."));
        doEmbed();
    }
}

void MainWindow::doEmbed()
{
    const QString text = descEdit->toPlainText().trimmed();
    if (text.isEmpty() || !modelLoaded)
        return;
    embedBtn->setEnabled(false);
    statusLabel->setText(S(SEmbedding));
    emit requestEmbed(text);
}

void MainWindow::onEmbedFinished(const QString &resultJson)
{
    embedBtn->setEnabled(true);
    const QJsonObject o = QJsonDocument::fromJson(resultJson.toUtf8()).object();
    if (!o.value("ok").toBool()) {
        onFailed(o.value("error").toString());
        return;
    }

    AnchorEntry e;
    const QJsonArray anchor = o.value("anchor").toArray();
    e.anchor.reserve(anchor.size());
    for (const auto &v : anchor)
        e.anchor.append(v.toDouble());
    e.gamesJson = QJsonDocument(o.value("games").toArray())
                      .toJson(QJsonDocument::Compact);
    e.tagsJson = QJsonDocument(o.value("tags").toArray())
                     .toJson(QJsonDocument::Compact);
    entries.append(e);

    const int row = anchorTable->rowCount();
    anchorTable->insertRow(row);
    // default anchor names are intentionally English in every UI language
    auto *nameItem = new QTableWidgetItem(
        QStringLiteral("Anchor %1").arg(row + 1));
    nameItem->setFlags(nameItem->flags() | Qt::ItemIsEditable);
    const QString preview = QStringLiteral("%1, %2, %3, …")
                                .arg(e.anchor.value(0), 0, 'f', 4)
                                .arg(e.anchor.value(1), 0, 'f', 4)
                                .arg(e.anchor.value(2), 0, 'f', 4);
    auto *vecItem = new QTableWidgetItem(preview);
    vecItem->setFlags(vecItem->flags() & ~Qt::ItemIsEditable);
    anchorTable->setItem(row, 0, nameItem);
    anchorTable->setItem(row, 1, vecItem);
    anchorTable->setCurrentCell(row, 0);
    exportBtn->setEnabled(true);

    showPredictions(e.gamesJson, e.tagsJson);
    statusLabel->setText(S(SDoneFmt)
                             .arg(o.value("n_sentences").toInt())
                             .arg(row + 1));
}

void MainWindow::showPredictions(const QString &gamesJson, const QString &tagsJson)
{
    const QJsonArray games = QJsonDocument::fromJson(gamesJson.toUtf8()).array();
    gameTable->setRowCount(0);
    for (const auto &g : games) {
        const QJsonArray pair = g.toArray();
        const int r = gameTable->rowCount();
        gameTable->insertRow(r);
        gameTable->setItem(r, 0, new QTableWidgetItem(pair.at(0).toString()));
        gameTable->setItem(r, 1, new QTableWidgetItem(
            QString::number(pair.at(1).toDouble(), 'f', 4)));
    }

    const QJsonArray tags = QJsonDocument::fromJson(tagsJson.toUtf8()).array();
    tagTable->setRowCount(0);
    for (const auto &t : tags) {
        const QJsonArray tri = t.toArray();
        const int r = tagTable->rowCount();
        tagTable->insertRow(r);
        auto *nameItem = new QTableWidgetItem(tri.at(0).toString());
        auto *probItem = new QTableWidgetItem(
            QString::number(tri.at(1).toDouble() * 100.0, 'f', 1) + QStringLiteral(" %"));
        if (tri.at(2).toBool()) {                       // above decision threshold
            QFont f = nameItem->font();
            f.setBold(true);
            nameItem->setFont(f);
            probItem->setFont(f);
        }
        tagTable->setItem(r, 0, nameItem);
        tagTable->setItem(r, 1, probItem);
    }
}

QString MainWindow::anchorRowsJson() const
{
    QJsonArray rows;
    for (int i = 0; i < entries.size(); ++i) {
        const QTableWidgetItem *nameItem = anchorTable->item(i, 0);
        QJsonArray vec;
        for (double v : entries[i].anchor)
            vec.append(v);
        QJsonArray row;
        row.append(nameItem ? nameItem->text()
                            : QStringLiteral("Anchor %1").arg(i + 1));
        row.append(vec);
        rows.append(row);
    }
    return QString::fromUtf8(QJsonDocument(rows).toJson(QJsonDocument::Compact));
}

void MainWindow::doExport()
{
    if (entries.isEmpty())
        return;
    const QString csvFilter = QStringLiteral("CSV (*.csv)");
    const QString jsonFilter = QStringLiteral("JSON (*.json)");
    QString selected;
    QString path = QFileDialog::getSaveFileName(
        this, S(SExportDialogTitle), QStringLiteral("anchors.csv"),
        csvFilter + QStringLiteral(";;") + jsonFilter, &selected);
    if (path.isEmpty())
        return;
    const bool json = selected == jsonFilter
                      || path.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive);
    if (!path.contains(QLatin1Char('.')))
        path += json ? QStringLiteral(".json") : QStringLiteral(".csv");

    emit requestExport(anchorRowsJson(), path,
                       json ? QStringLiteral("json") : QStringLiteral("csv"));
}

void MainWindow::anchorContextMenu(const QPoint &pos)
{
    const int row = anchorTable->rowAt(pos.y());
    if (row < 0 || row >= entries.size())
        return;

    const auto vecText = [this, row] {
        QStringList parts;
        for (double v : entries[row].anchor)
            parts << QString::number(v, 'f', 6);
        return QStringLiteral("[") + parts.join(QStringLiteral(", "))
               + QStringLiteral("]");
    };

    const QTableWidgetItem *nameItem = anchorTable->item(row, 0);
    const QString name = nameItem ? nameItem->text()
                                  : QStringLiteral("Anchor %1").arg(row + 1);

    QMenu menu(this);
    QAction *del = menu.addAction(QStringLiteral("Delete"));
    QAction *cpJson = menu.addAction(QStringLiteral("Copy as JSON"));
    QAction *cpPy = menu.addAction(QStringLiteral("Copy as PYTHON_LIST"));
    QAction *chosen = menu.exec(anchorTable->viewport()->mapToGlobal(pos));
    if (chosen == del) {
        deleteAnchorRow(row);
    } else if (chosen == cpJson) {
        // {"name": [dims...]} — same shape as the JSON export, one entry
        QJsonArray vec;
        for (double v : entries[row].anchor)
            vec.append(v);
        QJsonObject obj;
        obj.insert(name, vec);
        QGuiApplication::clipboard()->setText(QString::fromUtf8(
            QJsonDocument(obj).toJson(QJsonDocument::Compact)));
    } else if (chosen == cpPy) {
        QGuiApplication::clipboard()->setText(
            QStringLiteral("%1 = %2").arg(name.simplified()
                                              .replace(QLatin1Char(' '),
                                                       QLatin1Char('_')),
                                          vecText()));
    }
}

void MainWindow::deleteAnchorRow(int row)
{
    entries.removeAt(row);
    anchorTable->removeRow(row);
    if (entries.isEmpty()) {
        gameTable->setRowCount(0);
        tagTable->setRowCount(0);
        exportBtn->setEnabled(false);
    } else {
        const int cur = qBound(0, row, int(entries.size()) - 1);
        anchorTable->setCurrentCell(cur, 0);
        showPredictions(entries[cur].gamesJson, entries[cur].tagsJson);
    }
}

void MainWindow::onExportFinished(const QString &resultJson)
{
    const QJsonObject o = QJsonDocument::fromJson(resultJson.toUtf8()).object();
    if (!o.value("ok").toBool()) {
        onFailed(o.value("error").toString());
        return;
    }
    statusLabel->setText(S(SExportedFmt)
                             .arg(o.value("rows").toInt())
                             .arg(o.value("path").toString()));
}

void MainWindow::onFailed(const QString &error)
{
    embedBtn->setEnabled(modelLoaded);
    statusLabel->setText(S(SErrorStatus));
    QMessageBox::warning(this, S(SErrorTitle), error);
}
