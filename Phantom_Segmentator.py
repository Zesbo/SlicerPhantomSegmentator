import logging
import time
import os
import math
import json
import re
from collections import deque

import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
import SegmentStatistics

# ═════════════════════════════════════════════════════════════════════════════
# Specimen ID + specimen log (shared text node across modules)
# Keep this block identical in FemurROI, SimDXA_Femur and Phantom_Segmentator.
# ═════════════════════════════════════════════════════════════════════════════

SPECIMEN_ID_SEARCH = r"KS_[A-Z]+_\d+(?:_[LR])?|KS-\d+-\d+"  # e.g. KS_FM_02_R (femur), KS-05-26 (spine)
FEMUR_ID_PATTERN = r"KS_FM_\d+_[LR]"                 # KS = body donor, FM = femur, donor no., side
SPECIMEN_LOG_ATTRIBUTE = "SpecimenLog.SpecimenID"


def specimen_id_from_name(name):
    """Specimen ID contained in a node name ('KS_FM_02_R_CT_std_vBMD' -> 'KS_FM_02_R'), else ''."""
    m = re.search(SPECIMEN_ID_SEARCH, name or "")
    return m.group(0) if m else ""


def is_femur_specimen_id(specimen_id):
    return re.fullmatch(FEMUR_ID_PATTERN, specimen_id or "") is not None


def specimen_log_node(specimen_id, create=True):
    """Text node '<ID>_Log' shared by all modules (found via attribute, created on demand)."""
    if not specimen_id:
        return None
    for node in slicer.util.getNodesByClass("vtkMRMLTextNode"):
        if node.GetAttribute(SPECIMEN_LOG_ATTRIBUTE) == specimen_id:
            return node
    if not create:
        return None
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", f"{specimen_id}_Log")
    node.SetAttribute(SPECIMEN_LOG_ATTRIBUTE, specimen_id)
    return node


def log_specimen(specimen_id, module, text):
    """Append '<timestamp> [module] text' to the specimen log. Never raises."""
    try:
        node = specimen_log_node(specimen_id)
        if node is None or not text:
            return
        line = " | ".join(s.strip() for s in str(text).splitlines() if s.strip())
        node.SetText((node.GetText() or "") + f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{module}] {line}\n")
    except Exception as exc:
        logging.warning(f"[{module}] Specimen log failed: {exc}")


def write_specimen_log(specimen_id, output_dir):
    """Write the specimen log to <output_dir>/<ID>_log.txt. Returns the path or None."""
    node = specimen_log_node(specimen_id, create=False)
    if node is None or not output_dir:
        return None
    path = os.path.join(output_dir, f"{specimen_id}_log.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(node.GetText() or "")
    return path


def add_log_note_row(layout, get_specimen_id, module, set_status=None):
    """Line edit 'Log note' -> Enter appends 'NOTE: ...' to the specimen log (manual steps, observations)."""
    edit = qt.QLineEdit()
    edit.setPlaceholderText("Manual step or observation, Enter = add to specimen log")
    edit.setToolTip("Adds a timestamped NOTE line to the shared specimen log (<ID>_Log text node).")

    def _add():
        text = edit.text.strip()
        if not text:
            return
        sid = get_specimen_id()
        if not sid:
            slicer.util.warningDisplay("No specimen ID - note not logged.", windowTitle=module)
            return
        log_specimen(sid, module, "NOTE: " + text)
        edit.clear()
        if set_status:
            set_status(f"Note added to {sid}_Log.")

    edit.returnPressed.connect(_add)
    if isinstance(layout, qt.QFormLayout):
        layout.addRow("Log note:", edit)
    else:
        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel("Log note:"))
        row.addWidget(edit, 1)
        layout.addLayout(row)
    return edit

# Rods: voxels outside the reconstruction circle carry padding values (GE: -3024 HU) and are not data.
PADDING_HU_BELOW = -1500.0
ROD_MIN_COVERAGE = 0.5      # rod used for the regression only if >= 50 % of its cylinder lies in real data
ROD_MIN_FOR_REGRESSION = 3


class Phantom_Segmentator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Phantom Segmentator")
        self.parent.categories = ["Zesbo"]
        self.parent.dependencies = []
        self.parent.helpText = _(
            "Semi-automatic segmentation of CT calibration phantoms "
            "(BDC-6H200 reference, 3Dprinted_6H100_V1, uCT_4r_V2, uCT_6H_V1). "
            "Workflow: Place Seeds -> Segment Phantom -> Regression -> vBMD Volume."
        )
        self.parent.acknowledgementText = _("University of Leipzig")


class Phantom_SegmentatorWidget(ScriptedLoadableModuleWidget):

    # (ComboBox display name, geometry file name)
    PHANTOM_CONFIG = [
        ("BDC-6H200 (Reference)", "PHANTOM_GEOMETRY_BDC-6H200.json"),
        ("BDC-6H200 (Reference, FOV cut: 600/200)", "PHANTOM_GEOMETRY_BDC-6H200_FOVCUT.json"),
        ("uCT_4r_V2",             "PHANTOM_GEOMETRY_uCT_4r_V2.json"),
        ("3Dprinted_6H100_V1",    "PHANTOM_GEOMETRY_3Dprinted_6H100_V1.json"),
        ("uCT_6H_V1",             "PHANTOM_GEOMETRY_uCT_6H_V1.json"),
    ]

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # internal state variables
        self.currentVolumeNode = None
        self.currentPhantomGeometry = None
        self.seedMarkupsNode = None
        self._seedObserverTag = None
        self._seedRequiredCount = 0
        self._markupLine1 = None
        self._markupLine2 = None
        self._markupObserverTag = None
        self._markupPhase = 0
        self._phantomTransform_best = None
        self._phantomTransform_other = None
        self._phantomTransform_bestErr = None
        self._phantomTransform_otherErr = None
        self._phantomTransform_bestIsFlipped = None
        self._sceneObserverTags = []

        self._buildUI()
        self._connectSceneObservers()
        self._refreshVolumeCombo()
        self._loadCurrentPhantomGeometry()
        self._updateStatusLabel()

    def cleanup(self):
        for tag in self._sceneObserverTags:
            slicer.mrmlScene.RemoveObserver(tag)
        self._sceneObserverTags = []

    # ─── UI setup (no .ui file) ───────────────────────────────────────────────

    def _buildUI(self):
        # ---- Section: Phantom Setup ----
        setupBox = ctk.ctkCollapsibleButton()
        setupBox.text = "Phantom Setup"
        self.layout.addWidget(setupBox)
        setupLayout = qt.QFormLayout(setupBox)

        self.volumeCombo = qt.QComboBox()
        setupLayout.addRow("CT Volume:", self.volumeCombo)
        self.volumeCombo.currentIndexChanged.connect(self.onVolumeComboChanged)

        self.phantomTypeCombo = qt.QComboBox()
        for label, _ in self.PHANTOM_CONFIG:
            self.phantomTypeCombo.addItem(label)
        setupLayout.addRow("Phantom type:", self.phantomTypeCombo)
        self.phantomTypeCombo.currentIndexChanged.connect(self.onPhantomTypeChanged)

        # ---- Section: Segmentation ----
        segBox = ctk.ctkCollapsibleButton()
        segBox.text = "Segmentation"
        self.layout.addWidget(segBox)
        segLayout = qt.QVBoxLayout(segBox)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        segLayout.addWidget(self.statusLabel)

        self.setSeedsButton = qt.QPushButton("Place Seeds")
        self.setSeedsButton.setToolTip(
            "Starts seed placement mode for the registration rods "
            "and sets the current volume as the 2D background."
        )
        segLayout.addWidget(self.setSeedsButton)
        self.setSeedsButton.clicked.connect(self.onButton_SetSeeds)

        self.segmentButton = qt.QPushButton("Segment Phantom")
        self.segmentButton.setEnabled(False)
        segLayout.addWidget(self.segmentButton)
        self.segmentButton.clicked.connect(self.onButton_SegmentPhantom)

        self.flipButton = qt.QPushButton("Flip (alternative orientation)")
        self.flipButton.setEnabled(False)
        self.flipButton.setToolTip(
            "Applies the alternative phantom orientation (flip of rod A). "
            "Clicking again switches back."
        )
        segLayout.addWidget(self.flipButton)
        self.flipButton.clicked.connect(self.onButton_Flip)

        self.removeVoidsButton = qt.QPushButton("Remove Voids")
        self.removeVoidsButton.setEnabled(False)
        self.removeVoidsButton.setToolTip(
            "Removes voxels with HU < -250 (air inclusions/voids) from all segments "
            "and grows this region by 2 voxels (margin grow, 26-connectivity). "
            "Only available for 3D-printed phantoms."
        )
        segLayout.addWidget(self.removeVoidsButton)
        self.removeVoidsButton.clicked.connect(self.onButton_RemoveVoids)

        # ---- Section: Analysis ----
        analysisBox = ctk.ctkCollapsibleButton()
        analysisBox.text = "Analysis"
        self.layout.addWidget(analysisBox)
        analysisLayout = qt.QVBoxLayout(analysisBox)

        self.regressionButton = qt.QPushButton("Regression / Statistics")
        self.regressionButton.setToolTip(
            "Phantoms with reference values (BDC-6H200, uCT_4r_V2, 3Dprinted_6H100_V1): weighted linear regression CaHA = m*HU + b. "
            "Other phantoms: segment statistics only."
        )
        analysisLayout.addWidget(self.regressionButton)
        self.regressionButton.clicked.connect(self.onButton_Regression)

        self.vbmdButton = qt.QPushButton("Create vBMD Volume")
        self.vbmdButton.setEnabled(False)
        self.vbmdButton.setToolTip(
            "Only available after regression (reference phantom). "
            "Creates a new volume with vBMD = m*HU + b."
        )
        analysisLayout.addWidget(self.vbmdButton)
        self.vbmdButton.clicked.connect(self.onButton_CreateVbmdVolume)

        add_log_note_row(analysisLayout, self._specimenId, "Phantom_Segmentator")

        self.layout.addStretch()

    # ─── Scene observers ──────────────────────────────────────────────────────

    def _connectSceneObservers(self):
        scene = slicer.mrmlScene
        self._sceneObserverTags.append(
            scene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent,   self._onSceneChanged)
        )
        self._sceneObserverTags.append(
            scene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, self._onSceneChanged)
        )

    def _onSceneChanged(self, caller, event):
        self._refreshVolumeCombo()

    # ─── Volume-ComboBox ──────────────────────────────────────────────────────

    def _refreshVolumeCombo(self):
        # Remember node ID of the currently selected entry
        prevId = None
        if self.volumeCombo.count > 0:
            prevId = self.volumeCombo.itemData(self.volumeCombo.currentIndex)

        self.volumeCombo.blockSignals(True)
        self.volumeCombo.clear()

        volumeNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        count = volumeNodes.GetNumberOfItems()

        if count == 0:
            volumeNodes.UnRegister(None)
            self.volumeCombo.addItem("<< no volume in scene >>")
            self.volumeCombo.setEnabled(False)
            self.volumeCombo.blockSignals(False)
            self.currentVolumeNode = None
            self._updateButtonStates()
            self._updateStatusLabel()
            return

        self.volumeCombo.setEnabled(True)
        restoreIdx = 0
        for i in range(count):
            node = volumeNodes.GetItemAsObject(i)
            self.volumeCombo.addItem(node.GetName())
            self.volumeCombo.setItemData(i, node.GetID())
            if node.GetID() == prevId:
                restoreIdx = i

        volumeNodes.UnRegister(None)
        self.volumeCombo.blockSignals(False)

        # Set index and update state manually (signals were blocked)
        self.volumeCombo.setCurrentIndex(restoreIdx)
        self.onVolumeComboChanged(restoreIdx)

    def onVolumeComboChanged(self, index):
        nodeId = self.volumeCombo.itemData(index)
        self.currentVolumeNode = None
        if nodeId:
            try:
                self.currentVolumeNode = slicer.mrmlScene.GetNodeByID(nodeId)
            except Exception:
                pass

        # Make volume immediately visible in 2D views
        self._setVolumeAs2DBackground(self.currentVolumeNode)
        self._updateButtonStates()
        logging.info(f"[Phantom_Segmentator] Volume selected: {nodeId}")

    def _setVolumeAs2DBackground(self, volumeNode):
        if volumeNode is None:
            return
        slicer.util.setSliceViewerLayers(background=volumeNode)

    # ─── Phantom type combo box ───────────────────────────────────────────────

    def onPhantomTypeChanged(self, index):
        self._resetState()
        self._loadCurrentPhantomGeometry()
        self._updateStatusLabel()

    def _loadCurrentPhantomGeometry(self):
        idx = self.phantomTypeCombo.currentIndex
        if idx < 0 or idx >= len(self.PHANTOM_CONFIG):
            self.currentPhantomGeometry = None
            return

        _, filename = self.PHANTOM_CONFIG[idx]
        geomPath = self.resourcePath(f"Geometry/{filename}")

        if not os.path.isfile(geomPath):
            logging.error(f"[Phantom_Segmentator] Geometry file not found: {geomPath}")
            self.currentPhantomGeometry = None
            return

        with open(geomPath, "r", encoding="utf-8") as f:
            self.currentPhantomGeometry = json.load(f)

        logging.info(
            f"[Phantom_Segmentator] Geometry loaded: {filename}, "
            f"rods: {list(self.currentPhantomGeometry['rods'].keys())}"
        )

    # ─── State ────────────────────────────────────────────────────────────────

    def _resetState(self):
        # Remove seed observer first, before the node is deleted
        if self._seedObserverTag is not None and self.seedMarkupsNode is not None:
            try:
                self.seedMarkupsNode.RemoveObserver(self._seedObserverTag)
            except Exception:
                pass
        self._seedObserverTag = None
        self._seedRequiredCount = 0

        if self.seedMarkupsNode is not None:
            if slicer.mrmlScene.IsNodePresent(self.seedMarkupsNode):
                slicer.mrmlScene.RemoveNode(self.seedMarkupsNode)
            self.seedMarkupsNode = None

        # Clean up markup line observers and nodes
        if self._markupObserverTag is not None:
            for node in (self._markupLine1, self._markupLine2):
                if node is not None:
                    try:
                        node.RemoveObserver(self._markupObserverTag)
                    except Exception:
                        pass
            self._markupObserverTag = None
        self._markupPhase = 0
        for attr in ("_markupLine1", "_markupLine2"):
            node = getattr(self, attr, None)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, attr, None)

        self._phantomTransform_best = None
        self._phantomTransform_other = None
        self._phantomTransform_bestErr = None
        self._phantomTransform_otherErr = None
        self._phantomTransform_bestIsFlipped = None

        for attr in ("m", "b", "R2", "sigma_vbmd"):
            if hasattr(self, attr):
                delattr(self, attr)

        self._updateButtonStates()

    def _specimenId(self):
        node = getattr(self, "currentVolumeNode", None)
        return specimen_id_from_name(node.GetName()) if node is not None else ""

    def _log(self, text):
        log_specimen(self._specimenId(), "Phantom_Segmentator", text)

    def _updateButtonStates(self):
        hasVolume       = self.currentVolumeNode is not None
        hasAltTransform = self._phantomTransform_other is not None
        hasCalibration  = hasattr(self, "m") and hasattr(self, "b")
        hasSeg          = slicer.mrmlScene.GetFirstNodeByName("Phantom_Segmentation") is not None
        isVoidPhantom   = (self.currentPhantomGeometry is not None
                           and self.currentPhantomGeometry.get(
                               "remove_voids", "phantom_theory" not in self.currentPhantomGeometry))

        self.setSeedsButton.setEnabled(hasVolume)
        self.segmentButton.setEnabled(hasVolume)
        self.flipButton.setEnabled(hasAltTransform)
        self.vbmdButton.setEnabled(hasCalibration)
        self.removeVoidsButton.setVisible(isVoidPhantom)
        self.removeVoidsButton.setEnabled(isVoidPhantom and hasVolume and hasSeg)

    def _updateStatusLabel(self):
        if self.currentPhantomGeometry is None:
            self.statusLabel.setText("Geometry file could not be loaded.")
            self.setSeedsButton.setText("Place Seeds")
            return

        all_rods   = list(self.currentPhantomGeometry.get("rods", {}).keys())
        pt_label   = self.PHANTOM_CONFIG[self.phantomTypeCombo.currentIndex][0]
        has_theory = "phantom_theory" in self.currentPhantomGeometry
        strategy   = self.currentPhantomGeometry.get("registration_strategy", "axis_plus_midpoints")
        theory_str = "Reference values available (regression possible)" if has_theory \
                     else "No reference values (segment statistics only)"

        if strategy == "markup_lines":
            cfg  = self.currentPhantomGeometry.get("markup_lines_config", {})
            l1   = cfg.get("line1", {}); l2 = cfg.get("line2", {})
            reg_info = (
                "Line 1: " + l1.get("description", l1.get("label", "?")) + "\n"
                + "Line 2: " + l2.get("description", l2.get("label", "?"))
            )
            self.setSeedsButton.setText("Place Markups (2 lines)")
            self.setSeedsButton.setToolTip(
                "Starts markup line placement for the geometric registration "
                "of the uCT_4r_V2 phantom (no region growing)."
            )
        else:
            reg_rods = self.currentPhantomGeometry.get("registration_rods", [])
            reg_info = "Seeds for: " + ", ".join(f"Rod {r}" for r in reg_rods)
            self.setSeedsButton.setText("Place Seeds")
            self.setSeedsButton.setToolTip(
                "Starts seed placement mode for the registration rods "
                "and sets the current volume as the 2D background."
            )

        self.statusLabel.setText(
            f"Phantom type: {pt_label}\n"
            f"Rods: {', '.join(all_rods)}\n"
            f"{reg_info}\n"
            f"{theory_str}"
        )

    # ─── Place seeds / markup lines ───────────────────────────────────────────

    def onButton_SetSeeds(self):
        if self.currentPhantomGeometry is None:
            slicer.util.errorDisplay("No phantom geometry loaded.")
            return
        if self.currentVolumeNode is None:
            slicer.util.errorDisplay("No CT volume selected.")
            return

        strategy = self.currentPhantomGeometry.get("registration_strategy", "axis_plus_midpoints")
        if strategy == "markup_lines":
            self._startMarkupLinePlacement()
        else:
            self._startSeedPlacement()

    def _startSeedPlacement(self):
        reg_rods = self.currentPhantomGeometry.get("registration_rods", [])
        if len(reg_rods) < 2:
            slicer.util.errorDisplay("Geometry must contain at least 2 registration rods.")
            return

        logging.info(f"[Phantom_Segmentator] Seed mode started for rods: {reg_rods}")

        if self.seedMarkupsNode is not None and slicer.mrmlScene.IsNodePresent(self.seedMarkupsNode):
            slicer.mrmlScene.RemoveNode(self.seedMarkupsNode)
            self.seedMarkupsNode = None

        seedName = "PhantomSeeds_" + "_".join(reg_rods)
        self.seedMarkupsNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", seedName
        )

        self._setVolumeAs2DBackground(self.currentVolumeNode)

        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.SetActiveListID(self.seedMarkupsNode)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(1)

        # PointPositionDefinedEvent (not PointAddedEvent): ignore preview points
        self._seedRequiredCount = len(reg_rods)
        self._seedObserverTag = self.seedMarkupsNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self._onSeedPointAdded
        )

        lines = "\n".join(f"{i + 1}) in Rod {r}" for i, r in enumerate(reg_rods))
        hint  = self.currentPhantomGeometry.get("seed_hint", "")
        slicer.util.infoDisplay(
            f"Please place {len(reg_rods)} points:\n{lines}"
            + (f"\n\n{hint}" if hint else "")
            + "\n\nPlacement mode ends automatically after the last seed.",
            windowTitle="Phantom Segmentator"
        )

    def _startMarkupLinePlacement(self):
        cfg      = self.currentPhantomGeometry.get("markup_lines_config", {})
        l1_cfg   = cfg.get("line1", {}); l2_cfg = cfg.get("line2", {})
        l1_label = l1_cfg.get("label", "PhantomLine1")
        l1_desc  = l1_cfg.get("description", "First markup line")
        l2_desc  = l2_cfg.get("description", "Second markup line")

        # Remove old lines without observer call (observer may already be gone)
        self._markupObserverTag = None
        self._markupPhase = 0
        for attr in ("_markupLine1", "_markupLine2"):
            node = getattr(self, attr, None)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, attr, None)

        self._setVolumeAs2DBackground(self.currentVolumeNode)

        self._markupLine1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", l1_label)
        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.SetActiveListID(self._markupLine1)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(1)

        self._markupPhase = 1
        self._markupObserverTag = self._markupLine1.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self._onMarkupLinePointAdded
        )

        logging.info("[Phantom_Segmentator] Markup line mode: place line 1.")
        slicer.util.infoDisplay(
            f"Step 1 of 2:\n{l1_desc}\n\n"
            f"Then automatically step 2:\n{l2_desc}\n\n"
            "Please click both endpoints into the volume.",
            windowTitle="Phantom Segmentator - Markup Placement"
        )

    def _startLine2Placement(self):
        cfg      = self.currentPhantomGeometry.get("markup_lines_config", {})
        l2_cfg   = cfg.get("line2", {})
        l2_label = l2_cfg.get("label", "PhantomLine2")

        self._markupLine2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", l2_label)
        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.SetActiveListID(self._markupLine2)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(1)

        self._markupObserverTag = self._markupLine2.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self._onMarkupLinePointAdded
        )
        logging.info("[Phantom_Segmentator] Markup line mode: place line 2.")

    def _onMarkupLinePointAdded(self, caller, _event):
        n_defined = sum(
            1 for i in range(caller.GetNumberOfControlPoints())
            if caller.GetNthControlPointPositionStatus(i) == slicer.vtkMRMLMarkupsNode.PositionDefined
        )
        if n_defined < 2:
            return

        if self._markupObserverTag is not None:
            try:
                caller.RemoveObserver(self._markupObserverTag)
            except Exception:
                pass
            self._markupObserverTag = None

        if self._markupPhase == 1:
            # Line 1 done: briefly stop placement, then start line 2
            self._markupPhase = 2
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            qt.QTimer.singleShot(150, self._startLine2Placement)
            logging.info("[Phantom_Segmentator] Line 1 complete - starting line 2.")
        elif self._markupPhase == 2:
            # Line 2 done: end placement
            self._markupPhase = 0
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            interactionNode.SetPlaceModePersistence(0)
            logging.info("[Phantom_Segmentator] Both markup lines placed completely.")

    def _onSeedPointAdded(self, caller, _event):
        # Count only definitively placed points (ignore preview points)
        n = sum(
            1 for i in range(caller.GetNumberOfControlPoints())
            if caller.GetNthControlPointPositionStatus(i) == slicer.vtkMRMLMarkupsNode.PositionDefined
        )
        if n >= self._seedRequiredCount:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            interactionNode.SetPlaceModePersistence(0)
            if self._seedObserverTag is not None:
                try:
                    caller.RemoveObserver(self._seedObserverTag)
                except Exception:
                    pass
                self._seedObserverTag = None
            logging.info(
                f"[Phantom_Segmentator] {n}/{self._seedRequiredCount} seeds defined - "
                "placement mode ended automatically."
            )

    # ─── Main segmentation pipeline ───────────────────────────────────────────

    def onButton_SegmentPhantom(self):
        logging.info("[Phantom_Segmentator] Starting phantom segmentation.")
        t_total = time.perf_counter()

        volumeNode = self.currentVolumeNode
        if volumeNode is None:
            slicer.util.errorDisplay("No valid volume selected.")
            return

        if self.currentPhantomGeometry is None:
            slicer.util.errorDisplay("No phantom geometry loaded.")
            return

        strategy    = self.currentPhantomGeometry.get("registration_strategy", "axis_plus_midpoints")
        rodsPhantom = self.currentPhantomGeometry["rods"]

        # ── Step 1: phantom->CT transform (strategy-dependent) ──
        if strategy == "markup_lines":
            transform = self._getTransformFromMarkupLines()
        else:
            transform = self._getTransformFromSeeds(volumeNode)

        if transform is None:
            return

        # ── Step 2: cylinder segmentation ──
        segNames = list(rodsPhantom.keys())
        segNode  = self._getOrCreatePhantomSegmentationNode(volumeNode)
        self._ensureSegmentsExist(segNode, segNames)

        t0 = time.perf_counter()
        self._fillRodSegmentsWithCylinders(volumeNode, segNode, self.currentPhantomGeometry, transform)
        logging.info(f"[TIMING] Cylinder fill: {(time.perf_counter()-t0):.3f} s")
        logging.info(f"[TIMING] Total: {(time.perf_counter()-t_total):.3f} s")
        logging.info("[Phantom_Segmentator] Segmentation complete.")

        self.flipButton.setEnabled(self._phantomTransform_other is not None)
        self._updateButtonStates()

        cov = getattr(self, "_rodCoverage", {})
        low = [f"{k} {100 * v:.0f} %" for k, v in cov.items() if v < ROD_MIN_COVERAGE]
        covMsg = (f"\nRods outside the reconstructed volume (excluded from the regression): {', '.join(low)}"
                  if low else "")
        self._log(f"Phantom segmentation on {volumeNode.GetName()}: RMS {self._phantomTransform_bestErr:.3f} mm"
                  + (f"; low coverage: {', '.join(low)}" if low else ""))
        slicer.util.infoDisplay(
            f"Phantom segmentation complete.\n"
            f"Segments ({', '.join(segNames)}) in 'Phantom_Segmentation'.\n"
            f"RMS error (registration): {self._phantomTransform_bestErr:.3f} mm{covMsg}",
            windowTitle="Phantom Segmentator"
        )

    def _getTransformFromSeeds(self, volumeNode):
        """Registration via region growing (reference and 3Dprinted_6H100_V1 phantoms)."""
        reg_rods = self.currentPhantomGeometry.get("registration_rods", [])
        if len(reg_rods) < 2:
            slicer.util.errorDisplay("Geometry must define at least 2 registration rods.")
            return None

        if self.seedMarkupsNode is None or not slicer.mrmlScene.IsNodePresent(self.seedMarkupsNode):
            slicer.util.errorDisplay(
                f"No seeds found.\n"
                f"Please use 'Place Seeds' and place {len(reg_rods)} points "
                f"({', '.join(reg_rods)})."
            )
            return None

        n_defined = sum(
            1 for i in range(self.seedMarkupsNode.GetNumberOfControlPoints())
            if self.seedMarkupsNode.GetNthControlPointPositionStatus(i)
               == slicer.vtkMRMLMarkupsNode.PositionDefined
        )
        if n_defined < len(reg_rods):
            slicer.util.errorDisplay(
                f"Too few seeds: {n_defined} present, {len(reg_rods)} required "
                f"({', '.join(reg_rods)})."
            )
            return None

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        rodsPhantom = self.currentPhantomGeometry["rods"]
        seeds = {}
        for i, label in enumerate(reg_rods):
            pos = [0.0, 0.0, 0.0]
            self.seedMarkupsNode.GetNthControlPointPositionWorld(i, pos)
            seeds[label] = np.array(pos)
            logging.info(f"[Phantom_Segmentator] Seed rod {label}: {pos}")

        # Optional per-phantom region-growing settings (JSON key "seed_region_growing"):
        #   smoothing_radius_vox: box-mean filter radius before thresholding (1 -> 3x3x3), reduces noise
        #                         so rod and housing HU ranges no longer overlap (small rods, BONE kernel)
        #   roi_margin_mm:        growth limited to a sphere of (rod length + margin) around the seed
        #   rod_diameter_mm:      real rod diameter; enables the leak check
        #   max_volume_factor:    abort if the mask exceeds factor x the real rod volume (leak into housing)
        rgCfg = self.currentPhantomGeometry.get("seed_region_growing", {})

        rodEndpointsCT = {}
        rodCentersCT   = {}
        for label in reg_rods:
            rod    = rodsPhantom[label]
            thrMin = rod.get("hu_thr_min", 300)
            thrMax = rod.get("hu_thr_max", 1600)
            rodLen = float(np.linalg.norm(np.array(rod["p1_mm"], dtype=float) - np.array(rod["p0_mm"], dtype=float)))

            roiRadius = None
            if "roi_margin_mm" in rgCfg:
                roiRadius = rodLen + float(rgCfg["roi_margin_mm"])

            t0 = time.perf_counter()
            try:
                mask = self._segmentRodMaskFromSeed(
                    volumeNode, seeds[label], thrMin, thrMax,
                    smoothRadiusVox=int(rgCfg.get("smoothing_radius_vox", 0)),
                    roiRadiusMm=roiRadius)
            except RuntimeError as e:
                slicer.util.errorDisplay(f"Rod {label}: {e}", windowTitle="Phantom Segmentator")
                return None
            logging.info(f"[TIMING] Region-Growing Rod {label}: {(time.perf_counter()-t0)*1000:.1f} ms")

            # Leak check: mask volume vs. real rod volume
            if "rod_diameter_mm" in rgCfg:
                voxMm3   = float(np.prod(volumeNode.GetSpacing()))
                volMask  = float(mask.sum()) * voxMm3
                volRod   = math.pi * (0.5 * float(rgCfg["rod_diameter_mm"])) ** 2 * rodLen
                factor   = float(rgCfg.get("max_volume_factor", 1.6))
                logging.info(f"[Phantom_Segmentator] Rod {label}: mask {volMask:.0f} mm3, "
                             f"expected {volRod:.0f} mm3 (ratio {volMask / volRod:.2f})")
                if volMask > factor * volRod or volMask < 0.25 * volRod:
                    msg = (f"Region growing for rod {label} looks wrong: {volMask:.0f} mm³ found, "
                           f"~{volRod:.0f} mm³ expected (threshold [{thrMin}, {thrMax}] HU).\n"
                           + ("The mask leaked into the housing or a neighbouring rod. "
                              if volMask > factor * volRod else
                              "The mask is too small - seed outside the rod or threshold too narrow. ")
                           + "Check the seed position (centre of the rod) and the seed order.")
                    self._log(f"Region growing failed for {label}: {volMask:.0f} vs {volRod:.0f} mm3")
                    slicer.util.errorDisplay(msg, windowTitle="Phantom Segmentator")
                    return None

            center, _, p0RAS, p1RAS = self._computeRodCenterAxisAndEndpointsFromMask(volumeNode, mask)
            lenCT = float(np.linalg.norm(np.array(p1RAS) - np.array(p0RAS)))
            logging.info(f"[Phantom_Segmentator] Rod {label}: PCA length {lenCT:.1f} mm (nominal {rodLen:.1f} mm)")
            if abs(lenCT - rodLen) > 0.25 * rodLen:
                logging.warning(f"[Phantom_Segmentator] Rod {label}: PCA length deviates from nominal "
                                f"({lenCT:.1f} vs {rodLen:.1f} mm) - registration may be unreliable.")
            rodEndpointsCT[label] = (p0RAS, p1RAS)
            rodCentersCT[label]   = center

        t0 = time.perf_counter()
        result = self._computePhantomToCTTransform(
            self.currentPhantomGeometry, rodEndpointsCT, rodCentersCT
        )
        logging.info(f"[TIMING] Transform: {(time.perf_counter()-t0)*1000:.1f} ms")
        return result

    def _getTransformFromMarkupLines(self):
        """Registration via markup lines (uCT_4r_V2 phantom / markup_lines strategy)."""
        cfg = self.currentPhantomGeometry.get("markup_lines_config", {})

        def n_defined(node):
            if node is None or not slicer.mrmlScene.IsNodePresent(node):
                return 0
            return sum(
                1 for i in range(node.GetNumberOfControlPoints())
                if node.GetNthControlPointPositionStatus(i) == slicer.vtkMRMLMarkupsNode.PositionDefined
            )

        line1 = self._markupLine1
        line2 = self._markupLine2

        # Fallback: search by label (e.g. after phantom type change)
        if n_defined(line1) < 2:
            line1 = slicer.mrmlScene.GetFirstNodeByName(cfg.get("line1", {}).get("label", ""))
        if n_defined(line2) < 2:
            line2 = slicer.mrmlScene.GetFirstNodeByName(cfg.get("line2", {}).get("label", ""))

        if n_defined(line1) < 2 or n_defined(line2) < 2:
            slicer.util.errorDisplay(
                "Please place both markup lines completely first.\n"
                "Click 'Place Markups (2 lines)'."
            )
            return None

        return self._computeTransformFromMarkupLines(self.currentPhantomGeometry, line1, line2)

    def _computeTransformFromMarkupLines(self, phantomGeometry, line1Node, line2Node):
        """
        Rigid phantom->CT transform from 2 markup lines (uCT_4r_V2 phantom).

        Line 1: P_800 long axis - both endpoints in CT.
        Line 2: upper P_800 endpoint -> P_030 center (transverse orientation).

        Disambiguation: the line 1 endpoint closer to line 2's start point
        is identified as the upper phantom endpoint (p0_phantom_mm).
        3 non-collinear points define the rigid transform uniquely -
        no flip test required.
        """
        cfg    = phantomGeometry["markup_lines_config"]
        l1_cfg = cfg["line1"]
        l2_cfg = cfg["line2"]

        L1_A = [0.0]*3; line1Node.GetNthControlPointPositionWorld(0, L1_A)
        L1_B = [0.0]*3; line1Node.GetNthControlPointPositionWorld(1, L1_B)
        L2_A = [0.0]*3; line2Node.GetNthControlPointPositionWorld(0, L2_A)
        L2_B = [0.0]*3; line2Node.GetNthControlPointPositionWorld(1, L2_B)
        L1_A, L1_B, L2_A, L2_B = [np.array(p, dtype=float) for p in (L1_A, L1_B, L2_A, L2_B)]

        # Which endpoint of line 1 is the upper P_800 endpoint (= start of line 2)?
        if np.linalg.norm(L1_A - L2_A) <= np.linalg.norm(L1_B - L2_A):
            l1_p0_ct, l1_p1_ct = L1_A, L1_B
        else:
            l1_p0_ct, l1_p1_ct = L1_B, L1_A

        p0_ph = np.array(l1_cfg["p0_phantom_mm"], dtype=float)   # upper P_800 endpoint (phantom)
        p1_ph = np.array(l1_cfg["p1_phantom_mm"], dtype=float)   # lower P_800 endpoint (phantom)
        p2_ph = np.array(l2_cfg["p1_phantom_mm"], dtype=float)   # P_030 center (phantom)

        src = vtk.vtkPoints()
        tgt = vtk.vtkPoints()
        src.InsertNextPoint(p0_ph.tolist()); tgt.InsertNextPoint(l1_p0_ct.tolist())
        src.InsertNextPoint(p1_ph.tolist()); tgt.InsertNextPoint(l1_p1_ct.tolist())
        src.InsertNextPoint(p2_ph.tolist()); tgt.InsertNextPoint(L2_B.tolist())

        T = vtk.vtkLandmarkTransform()
        T.SetSourceLandmarks(src); T.SetTargetLandmarks(tgt)
        T.SetModeToRigidBody(); T.Update()

        err2 = 0.0
        for p_ph, p_ct in zip([p0_ph, p1_ph, p2_ph], [l1_p0_ct, l1_p1_ct, L2_B]):
            d = np.array(T.TransformPoint(p_ph.tolist()), dtype=float) - p_ct
            err2 += float(np.dot(d, d))
        rms = math.sqrt(err2 / 3.0)

        self._phantomTransform_best          = T
        self._phantomTransform_bestErr       = rms
        self._phantomTransform_other         = None
        self._phantomTransform_otherErr      = None
        self._phantomTransform_bestIsFlipped = False

        mat = T.GetMatrix()
        R   = np.array([[mat.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
        logging.info(
            f"[Phantom_Segmentator] Markup line transform: "
            f"det(R)={np.linalg.det(R):.4f}, RMS={rms:.4f} mm"
        )
        return T

    # ─── Region growing ───────────────────────────────────────────────────────

    @staticmethod
    def _boxMean(arr, r):
        """Box-mean filter with radius r voxels (window 2r+1) via cumulative sums; edges replicated."""
        if r <= 0:
            return arr
        w = 2 * r + 1
        a = np.pad(arr.astype(np.float64), r, mode="edge")
        for ax in range(3):
            c = np.cumsum(a, axis=ax)
            zero = np.zeros_like(np.take(c, [0], axis=ax))
            c = np.concatenate([zero, c], axis=ax)
            n = c.shape[ax]
            a = (np.take(c, range(w, n), axis=ax) - np.take(c, range(0, n - w), axis=ax)) / w
        return a

    def _segmentRodMaskFromSeed(self, volumeNode, seedRAS, thrMin, thrMax,
                                smoothRadiusVox=0, roiRadiusMm=None):
        logging.info(
            f"[Phantom_Segmentator] Region growing seed={list(seedRAS)}, "
            f"thr=[{thrMin}, {thrMax}], smoothing r={smoothRadiusVox} vox, roi={roiRadiusMm} mm"
        )

        fullArray     = slicer.util.arrayFromVolume(volumeNode)
        kMax, jMax, iMax = fullArray.shape

        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        ras = [float(seedRAS[0]), float(seedRAS[1]), float(seedRAS[2]), 1.0]
        ijk = [0.0, 0.0, 0.0, 0.0]
        rasToIjk.MultiplyPoint(ras, ijk)
        i0, j0, k0 = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))

        if not (0 <= i0 < iMax and 0 <= j0 < jMax and 0 <= k0 < kMax):
            raise RuntimeError("Seed point lies outside the volume extent.")

        # Optional crop to a box around the seed (ROI) and smoothing inside it
        off = (0, 0, 0)
        roiSq = None
        if roiRadiusMm is not None or smoothRadiusVox > 0:
            sp = volumeNode.GetSpacing()  # (i, j, k)
            rad = roiRadiusMm if roiRadiusMm is not None else 80.0
            ri, rj, rk = (int(math.ceil(rad / sp[0])) + 1, int(math.ceil(rad / sp[1])) + 1,
                          int(math.ceil(rad / sp[2])) + 1)
            ka, kb = max(0, k0 - rk), min(kMax, k0 + rk + 1)
            ja, jb = max(0, j0 - rj), min(jMax, j0 + rj + 1)
            ia, ib = max(0, i0 - ri), min(iMax, i0 + ri + 1)
            off = (ka, ja, ia)
            volArray = self._boxMean(fullArray[ka:kb, ja:jb, ia:ib], smoothRadiusVox)
            k0, j0, i0 = k0 - ka, j0 - ja, i0 - ia
            kMax, jMax, iMax = volArray.shape
            if roiRadiusMm is not None:
                roiSq = (roiRadiusMm ** 2, sp[0], sp[1], sp[2], k0, j0, i0)
        else:
            volArray = fullArray

        # If seed voxel is not within threshold: search a small neighborhood
        if not (thrMin <= volArray[k0, j0, i0] <= thrMax):
            found = False
            for dk in range(-2, 3):
                for dj in range(-2, 3):
                    for di in range(-2, 3):
                        kk, jj, ii = k0 + dk, j0 + dj, i0 + di
                        if 0 <= kk < kMax and 0 <= jj < jMax and 0 <= ii < iMax:
                            if thrMin <= volArray[kk, jj, ii] <= thrMax:
                                k0, j0, i0 = kk, jj, ii
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                raise RuntimeError(
                    f"No voxel within threshold [{thrMin},{thrMax}] found near seed."
                )

        visited = np.zeros_like(volArray, dtype=np.bool_)
        mask    = np.zeros_like(volArray, dtype=np.bool_)
        q       = deque([(k0, j0, i0)])
        visited[k0, j0, i0] = True
        maxVoxels = 500_000
        count = 0

        while q:
            k, j, i = q.popleft()
            if not (thrMin <= volArray[k, j, i] <= thrMax):
                continue
            mask[k, j, i] = True
            count += 1
            if count > maxVoxels:
                logging.warning(
                    "[Phantom_Segmentator] Region growing: voxel limit reached "
                    "(wrong seed or threshold too wide?)."
                )
                break
            for dk, dj, di in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                kk, jj, ii = k+dk, j+dj, i+di
                if 0 <= kk < kMax and 0 <= jj < jMax and 0 <= ii < iMax:
                    if not visited[kk, jj, ii]:
                        visited[kk, jj, ii] = True
                        if roiSq is not None:
                            r2, si, sj, sk, ck, cj, ci = roiSq
                            if ((ii - ci) * si) ** 2 + ((jj - cj) * sj) ** 2 + ((kk - ck) * sk) ** 2 > r2:
                                continue
                        if thrMin <= volArray[kk, jj, ii] <= thrMax:
                            q.append((kk, jj, ii))

        logging.info(f"[Phantom_Segmentator] Region growing: {count} voxels.")
        if off == (0, 0, 0) and mask.shape == fullArray.shape:
            return mask
        full = np.zeros(fullArray.shape, dtype=np.bool_)
        ka, ja, ia = off
        full[ka:ka + mask.shape[0], ja:ja + mask.shape[1], ia:ia + mask.shape[2]] = mask
        return full

    # ─── PCA axis determination ───────────────────────────────────────────────

    def _computeRodCenterAxisAndEndpointsFromMask(self, volumeNode, maskBool):
        indices = np.argwhere(maskBool)
        if indices.size == 0:
            raise RuntimeError("_computeRodCenterAxisAndEndpointsFromMask: mask is empty.")

        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)

        rasPoints = []
        for k, j, i in indices:
            ijk = [float(i), float(j), float(k), 1.0]
            ras = [0.0, 0.0, 0.0, 0.0]
            ijkToRas.MultiplyPoint(ijk, ras)
            rasPoints.append(ras[:3])

        pts    = np.array(rasPoints, dtype=float)
        center = pts.mean(axis=0)
        X      = pts - center
        cov    = X.T @ X / float(X.shape[0])

        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        axis = axis / np.linalg.norm(axis)

        s  = X @ axis
        p0 = center + s.min() * axis
        p1 = center + s.max() * axis

        return center, axis, p0, p1

    # ─── Phantom→CT landmark transform (generalized) ────────────────────────

    def _computePhantomToCTTransform(self, phantomGeometry, rodEndpointsCT, rodCentersCT=None):
        """
        Computes a rigid landmark transform phantom->CT.

        Two strategies (via 'registration_strategy' in the JSON):
        - 'axis_plus_midpoints' (default): rod A endpoints + rod B center, with flip test.
          Suitable for long, axis-parallel rods (reference, 3Dprinted_6H100_V1).
        - 'midpoints_only': centroids of all registration_rods as landmarks, no flip.
          Suitable for short / all-parallel arrangements (uCT_4r_V2).
        """
        strategy = phantomGeometry.get("registration_strategy", "axis_plus_midpoints")
        if strategy == "midpoints_only":
            centers = rodCentersCT if rodCentersCT is not None else {
                lbl: 0.5 * (np.array(rodEndpointsCT[lbl][0], dtype=float)
                            + np.array(rodEndpointsCT[lbl][1], dtype=float))
                for lbl in phantomGeometry["registration_rods"]
            }
            return self._computeTransformFromCenters(phantomGeometry, centers)

        reg_rods  = phantomGeometry["registration_rods"]
        rod_a_lbl = reg_rods[0]
        rod_b_lbl = reg_rods[1]
        rods_ph   = phantomGeometry["rods"]

        for lbl in reg_rods:
            if lbl not in rods_ph:
                raise RuntimeError(f"Rod '{lbl}' missing in phantom geometry.")
            if lbl not in rodEndpointsCT:
                raise RuntimeError(f"No CT endpoints for rod '{lbl}'.")

        def build_mapping(flip_a):
            mapping = {}

            # Rod A: axis rod; swap CT endpoints if needed (flip)
            r = rods_ph[rod_a_lbl]
            p0_ph = np.array(r["p0_mm"], dtype=float)
            p1_ph = np.array(r["p1_mm"], dtype=float)
            p0_ct = np.array(rodEndpointsCT[rod_a_lbl][0], dtype=float)
            p1_ct = np.array(rodEndpointsCT[rod_a_lbl][1], dtype=float)
            if flip_a:
                p0_ct, p1_ct = p1_ct, p0_ct
            mapping[rod_a_lbl] = (p0_ph, p1_ph, p0_ct, p1_ct)

            # Rod B: center rod; align axis with phantom axis
            r = rods_ph[rod_b_lbl]
            p0_ph = np.array(r["p0_mm"], dtype=float)
            p1_ph = np.array(r["p1_mm"], dtype=float)
            p0_ct = np.array(rodEndpointsCT[rod_b_lbl][0], dtype=float)
            p1_ct = np.array(rodEndpointsCT[rod_b_lbl][1], dtype=float)
            ax_ph = (p1_ph - p0_ph) / np.linalg.norm(p1_ph - p0_ph)
            ax_ct = (p1_ct - p0_ct)
            norm_ct = np.linalg.norm(ax_ct)
            if norm_ct > 1e-9:
                ax_ct /= norm_ct
                if np.dot(ax_ct, ax_ph) < 0.0:
                    p0_ct, p1_ct = p1_ct, p0_ct
            mapping[rod_b_lbl] = (p0_ph, p1_ph, p0_ct, p1_ct)

            return mapping

        def build_transform(mapping):
            src = vtk.vtkPoints()
            tgt = vtk.vtkPoints()

            # Rod A: both endpoints
            p0_ph_a, p1_ph_a, p0_ct_a, p1_ct_a = mapping[rod_a_lbl]
            src.InsertNextPoint(p0_ph_a.tolist())
            src.InsertNextPoint(p1_ph_a.tolist())
            tgt.InsertNextPoint(p0_ct_a.tolist())
            tgt.InsertNextPoint(p1_ct_a.tolist())

            # Rod B: center only
            p0_ph_b, p1_ph_b, p0_ct_b, p1_ct_b = mapping[rod_b_lbl]
            src.InsertNextPoint((0.5 * (p0_ph_b + p1_ph_b)).tolist())
            tgt.InsertNextPoint((0.5 * (p0_ct_b + p1_ct_b)).tolist())

            T = vtk.vtkLandmarkTransform()
            T.SetSourceLandmarks(src)
            T.SetTargetLandmarks(tgt)
            T.SetModeToRigidBody()
            T.Update()
            return T

        def rms_error(T, mapping):
            err2, n = 0.0, 0
            for lbl in reg_rods:
                p0_ph, p1_ph, p0_ct, p1_ct = mapping[lbl]
                for p_ph, p_ct in ((p0_ph, p0_ct), (p1_ph, p1_ct)):
                    d = np.array(T.TransformPoint(p_ph.tolist())) - p_ct
                    err2 += float(np.dot(d, d))
                    n    += 1
            return math.sqrt(err2 / n) if n > 0 else float("inf")

        # Variant normal
        map_n = build_mapping(flip_a=False)
        T_n   = build_transform(map_n)
        err_n = rms_error(T_n, map_n)

        # Variant flipped
        map_f = build_mapping(flip_a=True)
        T_f   = build_transform(map_f)
        err_f = rms_error(T_f, map_f)

        logging.info(
            f"[Phantom_Segmentator] Flip comparison rod '{rod_a_lbl}': "
            f"normal={err_n:.4f} mm, flipped={err_f:.4f} mm"
        )

        if err_f < err_n:
            chosen, err_c, other, err_o, is_flipped = T_f, err_f, T_n, err_n, True
        else:
            chosen, err_c, other, err_o, is_flipped = T_n, err_n, T_f, err_f, False

        self._phantomTransform_best          = chosen
        self._phantomTransform_bestErr       = err_c
        self._phantomTransform_other         = other
        self._phantomTransform_otherErr      = err_o
        self._phantomTransform_bestIsFlipped = is_flipped

        mat = chosen.GetMatrix()
        R   = np.array([[mat.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
        logging.info(
            f"[Phantom_Segmentator] Chosen: {'flipped' if is_flipped else 'normal'}, "
            f"det(R)={np.linalg.det(R):.4f}, RMS={err_c:.4f} mm"
        )

        return chosen

    def _computeTransformFromCenters(self, phantomGeometry, rodCentersCT):
        """
        Computes a rigid transform from the centroids of the registration rods.

        Suitable for phantoms with short or parallel-arranged rods (uCT_4r_V2):
        there the PCA axis determination yields no reliable endpoints, but the
        centroids can be determined robustly. With 3 non-collinear centroids the
        rigid transform is unique - no flip test needed.
        """
        reg_rods = phantomGeometry["registration_rods"]
        rods_ph  = phantomGeometry["rods"]

        for lbl in reg_rods:
            if lbl not in rods_ph:
                raise RuntimeError(f"Rod '{lbl}' missing in phantom geometry.")
            if lbl not in rodCentersCT:
                raise RuntimeError(f"No CT centroid for rod '{lbl}'.")

        src = vtk.vtkPoints()
        tgt = vtk.vtkPoints()
        for lbl in reg_rods:
            rod    = rods_ph[lbl]
            mid_ph = 0.5 * (np.array(rod["p0_mm"], dtype=float) + np.array(rod["p1_mm"], dtype=float))
            ctr_ct = np.array(rodCentersCT[lbl], dtype=float)
            src.InsertNextPoint(mid_ph.tolist())
            tgt.InsertNextPoint(ctr_ct.tolist())

        T = vtk.vtkLandmarkTransform()
        T.SetSourceLandmarks(src)
        T.SetTargetLandmarks(tgt)
        T.SetModeToRigidBody()
        T.Update()

        # RMS error over all centroids
        err2, n = 0.0, 0
        for lbl in reg_rods:
            mid_ph = 0.5 * (np.array(rods_ph[lbl]["p0_mm"]) + np.array(rods_ph[lbl]["p1_mm"]))
            ctr_ct = np.array(rodCentersCT[lbl])
            d   = np.array(T.TransformPoint(mid_ph.tolist())) - ctr_ct
            err2 += float(np.dot(d, d))
            n   += 1
        rms = math.sqrt(err2 / n) if n > 0 else 0.0

        self._phantomTransform_best          = T
        self._phantomTransform_bestErr       = rms
        self._phantomTransform_other         = None   # no flip with centroid strategy
        self._phantomTransform_otherErr      = None
        self._phantomTransform_bestIsFlipped = False

        mat = T.GetMatrix()
        R   = np.array([[mat.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
        logging.info(
            f"[Phantom_Segmentator] Centroid transform ({len(reg_rods)} rods): "
            f"det(R)={np.linalg.det(R):.4f}, RMS={rms:.4f} mm"
        )
        return T

    # ─── Segmentation node ────────────────────────────────────────────────────

    def _getOrCreatePhantomSegmentationNode(self, volumeNode):
        name = "Phantom_Segmentation"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
            logging.info("[Phantom_Segmentator] New segmentation node created.")
        node.CreateDefaultDisplayNodes()
        node.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        return node

    def _ensureSegmentsExist(self, segmentationNode, segmentNames):
        seg = segmentationNode.GetSegmentation()
        for name in segmentNames:
            segId = seg.GetSegmentIdBySegmentName(name)
            if not segId:
                newId = seg.AddEmptySegment()
                seg.GetSegment(newId).SetName(name)
                logging.info(f"[Phantom_Segmentator] Segment '{name}' created ({newId}).")

    # ─── Cylinder segmentation (vectorized, all rods from geometry) ────────────

    def _fillRodSegmentsWithCylinders(self, volumeNode, segmentationNode,
                                       phantomGeometry, phantomToCtTransform):
        logging.info("[Phantom_Segmentator] Starting cylinder segmentation (vectorized).")

        ijkToRas = vtk.vtkMatrix4x4()
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        extent      = volumeNode.GetImageData().GetExtent()
        rods_ph     = phantomGeometry["rods"]
        huAll       = slicer.util.arrayFromVolume(volumeNode)
        voxelMm3    = float(np.prod(volumeNode.GetSpacing()))
        self._rodCoverage = {label: 0.0 for label in rods_ph}
        segmentation = segmentationNode.GetSegmentation()

        # IJK->RAS matrix as numpy array (faster than repeated GetElement)
        mat = np.array(
            [[ijkToRas.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=float
        )

        for label, rod in rods_ph.items():
            # Determine radius (supports radius_mm and diameter_mm)
            if "radius_mm" in rod:
                radius_mm = float(rod["radius_mm"])
            elif "diameter_mm" in rod:
                radius_mm = float(rod["diameter_mm"]) / 2.0
            else:
                radius_mm = 8.0

            # Transform endpoints into CT-RAS
            p0_ras = np.array(phantomToCtTransform.TransformPoint(rod["p0_mm"]), dtype=float)
            p1_ras = np.array(phantomToCtTransform.TransformPoint(rod["p1_mm"]), dtype=float)

            v      = p1_ras - p0_ras
            length = float(np.linalg.norm(v))
            if length < 1e-3:
                logging.warning(f"[Phantom_Segmentator] Rod {label}: length ~0, skipping.")
                continue
            u = v / length

            # Axis-aligned bounding box (with radius padding)
            minX = min(p0_ras[0], p1_ras[0]) - radius_mm
            maxX = max(p0_ras[0], p1_ras[0]) + radius_mm
            minY = min(p0_ras[1], p1_ras[1]) - radius_mm
            maxY = max(p0_ras[1], p1_ras[1]) + radius_mm
            minZ = min(p0_ras[2], p1_ras[2]) - radius_mm
            maxZ = max(p0_ras[2], p1_ras[2]) + radius_mm

            corners = []
            for cx in (minX, maxX):
                for cy in (minY, maxY):
                    for cz in (minZ, maxZ):
                        h = [0.0, 0.0, 0.0, 0.0]
                        rasToIjk.MultiplyPoint([cx, cy, cz, 1.0], h)
                        corners.append(h)

            allI = [c[0] for c in corners]
            allJ = [c[1] for c in corners]
            allK = [c[2] for c in corners]

            iMin = max(extent[0], int(math.floor(min(allI))))
            iMax = min(extent[1], int(math.ceil(max(allI))))
            jMin = max(extent[2], int(math.floor(min(allJ))))
            jMax = min(extent[3], int(math.ceil(max(allJ))))
            kMin = max(extent[4], int(math.floor(min(allK))))
            kMax = min(extent[5], int(math.ceil(max(allK))))

            if iMin > iMax or jMin > jMax or kMin > kMax:
                logging.warning(f"[Phantom_Segmentator] Rod {label}: bounding box empty, skipping.")
                continue

            segmentId = segmentation.GetSegmentIdBySegmentName(label)
            if not segmentId:
                logging.warning(f"[Phantom_Segmentator] Segment '{label}' not found, skipping.")
                continue

            segArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
            if segArray is None:
                logging.warning(f"[Phantom_Segmentator] arrayFromSegmentBinaryLabelmap for '{label}' failed.")
                continue

            segArray[:] = 0

            # Vectorized cylinder test in the bounding-box subvolume
            K, J, I = np.meshgrid(
                np.arange(kMin, kMax + 1, dtype=float),
                np.arange(jMin, jMax + 1, dtype=float),
                np.arange(iMin, iMax + 1, dtype=float),
                indexing="ij"
            )
            x = mat[0,0]*I + mat[0,1]*J + mat[0,2]*K + mat[0,3]
            y = mat[1,0]*I + mat[1,1]*J + mat[1,2]*K + mat[1,3]
            z = mat[2,0]*I + mat[2,1]*J + mat[2,2]*K + mat[2,3]
            P = np.stack([x, y, z], axis=-1)

            w           = P - p0_ras
            s           = w[...,0]*u[0] + w[...,1]*u[1] + w[...,2]*u[2]
            mask_axial  = (s >= 0.0) & (s <= length)
            closest     = p0_ras + s[..., None] * u
            diff        = P - closest
            mask_radial = (diff[...,0]**2 + diff[...,1]**2 + diff[...,2]**2) <= radius_mm**2

            subArray = segArray[kMin:kMax+1, jMin:jMax+1, iMin:iMax+1]
            inCyl    = mask_axial & mask_radial
            realData = huAll[kMin:kMax+1, jMin:jMax+1, iMin:iMax+1] > PADDING_HU_BELOW
            subArray[inCyl & realData] = 1

            n_vox      = int((inCyl & realData).sum())
            n_expected = math.pi * radius_mm ** 2 * length / voxelMm3
            self._rodCoverage[label] = n_vox / n_expected if n_expected > 0 else 0.0
            logging.info(f"[Phantom_Segmentator] Rod {label}: {n_vox} voxels set "
                         f"(coverage {100 * self._rodCoverage[label]:.0f} %, "
                         f"{int((inCyl & ~realData).sum())} padding voxels skipped).")

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segArray, segmentationNode, segmentId, volumeNode
            )

        logging.info("[Phantom_Segmentator] Cylinder segmentation complete.")

    # ─── Regression / Statistics ──────────────────────────────────────────────

    def onButton_Regression(self):
        logging.info("[Phantom_Segmentator] Starting segment statistics.")

        volumeNode = self.currentVolumeNode
        if volumeNode is None:
            slicer.util.errorDisplay("No valid volume selected.")
            return

        segNode = slicer.mrmlScene.GetFirstNodeByName("Phantom_Segmentation")
        if segNode is None:
            slicer.util.errorDisplay(
                "Segmentation 'Phantom_Segmentation' not found.\n"
                "Please run 'Segment Phantom' first."
            )
            return

        # Compute segment statistics
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        paramNode    = segStatLogic.getParameterNode()
        paramNode.SetParameter("Segmentation",                              segNode.GetID())
        paramNode.SetParameter("ScalarVolume",                              volumeNode.GetID())
        paramNode.SetParameter("LabelmapSegmentStatisticsPlugin.enabled",   "False")
        paramNode.SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", "True")
        # Shared scene singleton (also used by the Segment Statistics module and FemurROI):
        # defaults only apply to unset keys, so enable the keys used below explicitly.
        paramNode.SetParameter("visibleSegmentsOnly", "False")
        for key in ("mean", "stdev", "voxel_count"):
            paramNode.SetParameter(f"ScalarVolumeSegmentStatisticsPlugin.{key}.enabled", "True")
        segStatLogic.computeStatistics()

        tableName = "Phantom_Segmentation_Statistics"
        tableNode = slicer.mrmlScene.GetFirstNodeByName(tableName)
        if tableNode is None:
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", tableName)
        segStatLogic.exportToTable(tableNode)
        logging.info("[Phantom_Segmentator] Segment statistics exported.")

        # Parse statistics table
        table    = tableNode.GetTable()
        colSeg   = table.GetColumn(table.GetColumnIndex("Segment"))
        colMean  = table.GetColumn(table.GetColumnIndex("Mean"))
        colStd   = table.GetColumn(table.GetColumnIndex("Standard deviation"))
        colCount = table.GetColumn(table.GetColumnIndex("Voxel count"))

        if colSeg is None or colMean is None:
            logging.error("[Phantom_Segmentator] Required columns ('Segment','Mean') missing in statistics table.")
            return

        segmentMean, segmentStd, segmentVoxelCount = {}, {}, {}
        for r in range(table.GetNumberOfRows()):
            name = colSeg.GetValue(r)
            try:
                segmentMean[name] = float(colMean.GetValue(r))
            except ValueError:
                continue
            if colStd is not None:
                try:
                    segmentStd[name] = float(colStd.GetValue(r))
                except ValueError:
                    segmentStd[name] = float("nan")
            if colCount is not None:
                try:
                    segmentVoxelCount[name] = float(colCount.GetValue(r))
                except ValueError:
                    segmentVoxelCount[name] = float("nan")

        logging.info(f"[Phantom_Segmentator] SegmentMean: {segmentMean}")

        # No reference values present (e.g. uCT_6H_V1 before cross-calibration): show statistics only
        if "phantom_theory" not in self.currentPhantomGeometry:
            hu_info = "\n".join(
                f"  Rod {k}: {v:.1f} HU" for k, v in segmentMean.items()
            )
            slicer.util.infoDisplay(
                "Segment statistics computed.\n"
                "No reference values for this phantom type (HU statistics only).\n\n"
                f"{hu_info}\n\nDetails in table '{tableName}'.",
                windowTitle="Phantom Segmentator"
            )
            return

        # ── Weighted linear regression for the reference phantom ──

        phantom_theory = self.currentPhantomGeometry["phantom_theory"]
        _volName = self.currentVolumeNode.GetName() if self.currentVolumeNode is not None else ""
        if "phantom_theory_bone" in self.currentPhantomGeometry and re.search(r"bone", _volName, re.IGNORECASE):
            phantom_theory = self.currentPhantomGeometry["phantom_theory_bone"]
            logging.info("[Phantom_Segmentator] Using BONEPLUS reference values (phantom_theory_bone).")
        # Rods in geometry-definition order, filtered to those with values
        coverage = getattr(self, "_rodCoverage", {})
        seg_order = [
            lbl for lbl in self.currentPhantomGeometry["rods"]
            if lbl in phantom_theory and lbl in segmentMean
            and coverage.get(lbl, 1.0) >= ROD_MIN_COVERAGE
        ]
        excluded = [lbl for lbl in self.currentPhantomGeometry["rods"]
                    if lbl in phantom_theory and lbl not in seg_order]

        if len(seg_order) < ROD_MIN_FOR_REGRESSION:
            msg = (f"Only {len(seg_order)} usable rods ({', '.join(seg_order) or '-'}) - at least "
                   f"{ROD_MIN_FOR_REGRESSION} required. Excluded: {', '.join(excluded) or '-'}.")
            logging.error("[Phantom_Segmentator] " + msg)
            self._log("Regression failed: " + msg)
            slicer.util.errorDisplay(msg, windowTitle="Phantom Segmentator")
            return

        x_vals       = np.array([segmentMean[s]                      for s in seg_order], dtype=float)
        y_vals       = np.array([phantom_theory[s]["caHA_mgcm3"]      for s in seg_order], dtype=float)
        sigma_ph     = np.array([phantom_theory[s]["sigma_mgcm3"]     for s in seg_order], dtype=float)

        sigma_hu = []
        for s in seg_order:
            std_v   = segmentStd.get(s, float("nan"))
            count_v = segmentVoxelCount.get(s, float("nan"))
            if (not math.isnan(std_v)) and (not math.isnan(count_v)) and count_v > 0:
                sigma_hu.append(std_v / math.sqrt(count_v))
            else:
                sigma_hu.append(0.0)
        sigma_hu = np.array(sigma_hu, dtype=float)

        # First unweighted pass for m_0 (needed for weight computation)
        m0, _ = np.polyfit(x_vals, y_vals, 1)

        sigma_y_sq = np.maximum(sigma_ph**2 + (abs(m0) * sigma_hu)**2, 1e-6)
        w_sqrt     = 1.0 / np.sqrt(sigma_y_sq)

        X  = np.column_stack([np.ones_like(x_vals), x_vals])
        Xw = X  * w_sqrt[:, None]
        yw = y_vals * w_sqrt

        try:
            beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
            b, m = beta[0], beta[1]
        except np.linalg.LinAlgError:
            logging.warning("[Phantom_Segmentator] Weighted regression singular - using unweighted.")
            m, b = m0, float(np.polyfit(x_vals, y_vals, 1)[1])

        self.m = m
        self.b = b
        self.rods_used = list(seg_order)
        self.rods_excluded = list(excluded)

        y_fit    = m * x_vals + b
        residuals = y_vals - y_fit
        ss_res   = float(np.sum(residuals**2))
        ss_tot   = float(np.sum((y_vals - np.mean(y_vals))**2))
        self.R2  = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        dof      = max(1, len(x_vals) - 2)
        self.sigma_vbmd = math.sqrt(ss_res / dof)

        logging.info(
            f"[Phantom_Segmentator] Regression: m={m:.6f}, b={b:.4f}, "
            f"R2={self.R2:.6f}, sigma={self.sigma_vbmd:.4f} mgCaHA/cm3"
        )
        print(
            f"[Phantom_Segmentator] CaHA = {m:.6f} * HU + {b:.4f}  "
            f"R2={self.R2:.6f}  sigma={self.sigma_vbmd:.4f} mgCaHA/cm3"
        )

        # Error metrics
        abs_err = np.abs(residuals)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(y_vals != 0.0, abs_err / y_vals * 100.0, np.nan)

        sigma_y_hu    = abs(m) * sigma_hu
        sigma_total   = np.sqrt(sigma_ph**2 + sigma_y_hu**2 + self.sigma_vbmd**2)

        # Build error table
        errTableName = "Phantom_Regression_Error"
        errTableNode = slicer.mrmlScene.GetFirstNodeByName(errTableName)
        if errTableNode is None:
            errTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", errTableName)
        errTable = errTableNode.GetTable()
        errTable.Initialize()

        def _arr(cls, colName):
            c = cls()
            c.SetName(colName)
            return c

        cols = [
            _arr(vtk.vtkStringArray,  "Segment"),
            _arr(vtk.vtkDoubleArray,  "vBMD_theory_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "HU_mean"),
            _arr(vtk.vtkDoubleArray,  "vBMD_fit_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "AbsError_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "RelError_percent"),
            _arr(vtk.vtkDoubleArray,  "Sigma_phantom_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "Sigma_HU_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "Sigma_resid_mgCaHA_cm3"),
            _arr(vtk.vtkDoubleArray,  "Sigma_total_mgCaHA_cm3"),
        ]
        segCol, theoryCol, huCol, fitCol, absCol, relCol, sigPhCol, sigHuCol, sigRCol, sigTotCol = cols

        for i in range(len(seg_order)):
            segCol.InsertNextValue(seg_order[i])
            theoryCol.InsertNextValue(float(y_vals[i]))
            huCol.InsertNextValue(float(x_vals[i]))
            fitCol.InsertNextValue(float(y_fit[i]))
            absCol.InsertNextValue(float(abs_err[i]))
            relCol.InsertNextValue(float(rel_err[i]))
            sigPhCol.InsertNextValue(float(sigma_ph[i]))
            sigHuCol.InsertNextValue(float(sigma_y_hu[i]))
            sigRCol.InsertNextValue(float(self.sigma_vbmd))
            sigTotCol.InsertNextValue(float(sigma_total[i]))

        for col in cols:
            errTable.AddColumn(col)

        logging.info("[Phantom_Segmentator] Error table 'Phantom_Regression_Error' updated.")
        self._updateButtonStates()

        exclMsg = (f"\nExcluded rods (outside the volume): {', '.join(excluded)}\n"
                   f"Calibration range {y_vals.min():.0f}-{y_vals.max():.0f} mgCaHA/cm³ - values outside are "
                   "extrapolated." if excluded else "")
        volName = self.currentVolumeNode.GetName() if self.currentVolumeNode is not None else "?"
        self._log(f"Regression on {volName}: CaHA = {m:.5f} * HU + {b:.4f}, R2 = {self.R2:.5f}, "
                  f"sigma = {self.sigma_vbmd:.3f} mgCaHA/cm3, {len(seg_order)} rods ({', '.join(seg_order)})"
                  + (f", excluded: {', '.join(excluded)}" if excluded else ""))
        slicer.util.infoDisplay(
            f"Regression complete.\n"
            f"CaHA = {m:.5f} · HU + {b:.4f}\n"
            f"R² = {self.R2:.5f}\n"
            f"σ = {self.sigma_vbmd:.3f} mgCaHA/cm³\n"
            f"Rods used: {len(seg_order)}{exclMsg}",
            windowTitle="Phantom Segmentator"
        )

    # ─── vBMD Volume ──────────────────────────────────────────────────────────

    def onButton_CreateVbmdVolume(self):
        logging.info("[Phantom_Segmentator] Creating vBMD volume.")

        if not (hasattr(self, "m") and hasattr(self, "b")):
            slicer.util.errorDisplay(
                "Calibration not computed yet.\n"
                "Please run 'Regression / Statistics' first."
            )
            return

        volumeNode = self.currentVolumeNode
        if volumeNode is None:
            slicer.util.errorDisplay("No valid volume selected.")
            return

        if re.search(r"bone", volumeNode.GetName(), re.IGNORECASE) and not slicer.util.confirmYesNoDisplay(
                f"'{volumeNode.GetName()}' looks like the BONEPLUS reconstruction.\n"
                "HU calibration and vBMD belong on the STANDARD kernel.\n\nCreate the vBMD volume anyway?",
                windowTitle="Phantom Segmentator"):
            return
        if re.search(r"bone", volumeNode.GetName(), re.IGNORECASE):
            self._log(f"WARNING: vBMD created from '{volumeNode.GetName()}' (BONEPLUS?) after confirmation")

        m, b    = float(self.m), float(self.b)
        newName = volumeNode.GetName() + "_vBMD"

        vbmdNode = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, volumeNode, newName)
        logging.info(f"[Phantom_Segmentator] vBMD volume '{newName}' created.")

        t0       = time.perf_counter()
        huArray  = slicer.util.arrayFromVolume(volumeNode).astype(np.float32)
        vbmdArray = m * huArray + b
        slicer.util.updateVolumeFromArray(vbmdNode, vbmdArray)
        logging.info(f"[Phantom_Segmentator] vBMD computed ({time.perf_counter()-t0:.3f} s).")

        # ── Calibration metadata (persists two ways) ──
        # 1) MRML attributes on the vBMD node -> survive in the .mrb scene,
        #    read back in-session by FemurROI.
        # 2) Sidecar JSON next to the source volume file -> survives
        #    save/reload of the raw volume outside any scene.
        try:
            phantomLabel = self.PHANTOM_CONFIG[self.phantomTypeCombo.currentIndex][0]
        except Exception:
            phantomLabel = "unknown"
        R2    = float(getattr(self, "R2", float("nan")))
        sigma = float(getattr(self, "sigma_vbmd", float("nan")))
        dateStr = time.strftime("%Y-%m-%d")

        meta = {
            "calib_m": m,
            "calib_b": b,
            "unit": "mgCaHA/cm3",
            "R2": R2,
            "sigma_vbmd_mgCaHA_cm3": sigma,
            "phantom": phantomLabel,
            "date": dateStr,
            "source_volume": volumeNode.GetName(),
            "module": "Phantom_Segmentator",
            "n_rods": len(getattr(self, "rods_used", [])) or None,
            "rods_used": ",".join(getattr(self, "rods_used", [])),
            "rods_excluded": ",".join(getattr(self, "rods_excluded", [])),
        }

        vbmdNode.SetAttribute("Phantom_Segmentator.m", str(m))
        vbmdNode.SetAttribute("Phantom_Segmentator.b", str(b))
        vbmdNode.SetAttribute("Phantom_Segmentator.unit", "mgCaHA/cm3")
        vbmdNode.SetAttribute("Phantom_Segmentator.R2", str(R2))
        vbmdNode.SetAttribute("Phantom_Segmentator.sigma_vbmd_mgCaHA_cm3", str(sigma))
        vbmdNode.SetAttribute("Phantom_Segmentator.phantom", phantomLabel)
        vbmdNode.SetAttribute("Phantom_Segmentator.date", dateStr)
        vbmdNode.SetAttribute("Phantom_Segmentator.source_volume", volumeNode.GetName())
        if meta["n_rods"]:
            vbmdNode.SetAttribute("Phantom_Segmentator.n_rods", str(meta["n_rods"]))
        vbmdNode.SetAttribute("Phantom_Segmentator.rods_used", meta["rods_used"])
        vbmdNode.SetAttribute("Phantom_Segmentator.rods_excluded", meta["rods_excluded"])

        sidecarMsg = ""
        try:
            srcStorage = volumeNode.GetStorageNode()
            srcPath = srcStorage.GetFileName() if srcStorage else None
            if srcPath and os.path.isdir(os.path.dirname(srcPath)):
                sidecarPath = os.path.join(os.path.dirname(srcPath), newName + "_calib.json")
                with open(sidecarPath, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2, ensure_ascii=False)
                vbmdNode.SetAttribute("Phantom_Segmentator.sidecar", sidecarPath)
                sidecarMsg = f"\nSidecar: {sidecarPath}"
                logging.info(f"[Phantom_Segmentator] Calib sidecar written: {sidecarPath}")
            else:
                sidecarMsg = ("\n(No source file on disk - sidecar skipped; "
                              "save the vBMD volume and re-run to write it.)")
        except Exception as exc:
            logging.warning(f"[Phantom_Segmentator] Sidecar write failed: {exc}")
            sidecarMsg = f"\n(Sidecar write failed: {exc})"

        slicer.util.infoDisplay(
            f"vBMD volume '{newName}' created.\n"
            f"vBMD = {m:.5f} · HU + {b:.4f}\n"
            f"Phantom: {phantomLabel} · R2={R2:.4f} · sigma={sigma:.2f} mgCaHA/cm3"
            f"{sidecarMsg}",
            windowTitle="Phantom Segmentator"
        )
        self._log(f"vBMD volume '{newName}' created from {volumeNode.GetName()} "
                  f"(m = {m:.5f}, b = {b:.4f}, {meta['n_rods']} rods){sidecarMsg}")
        logging.info("[Phantom_Segmentator] vBMD volume complete.")

    # ─── Flip ─────────────────────────────────────────────────────────────────

    def onButton_Flip(self):
        logging.info("[Phantom_Segmentator] Flip selected.")

        transform = self._phantomTransform_other
        if transform is None:
            slicer.util.errorDisplay(
                "No alternative transform available.\n"
                "Please run 'Segment Phantom' first.",
                windowTitle="Phantom Segmentator - Flip"
            )
            return

        volumeNode = self.currentVolumeNode
        if volumeNode is None:
            slicer.util.errorDisplay("No volume selected.", windowTitle="Phantom Segmentator - Flip")
            return

        if self.currentPhantomGeometry is None:
            slicer.util.errorDisplay("No phantom geometry.", windowTitle="Phantom Segmentator - Flip")
            return

        segNames = list(self.currentPhantomGeometry["rods"].keys())
        segNode  = self._getOrCreatePhantomSegmentationNode(volumeNode)
        self._ensureSegmentsExist(segNode, segNames)

        t0 = time.perf_counter()
        self._fillRodSegmentsWithCylinders(volumeNode, segNode, self.currentPhantomGeometry, transform)
        logging.info(
            f"[Phantom_Segmentator] Flip applied ({time.perf_counter()-t0:.3f} s), "
            f"RMS={self._phantomTransform_otherErr:.4f} mm."
        )

        # Swap variants -> clicking flip again switches back
        self._phantomTransform_best,     self._phantomTransform_other     = \
            self._phantomTransform_other, self._phantomTransform_best
        self._phantomTransform_bestErr,  self._phantomTransform_otherErr  = \
            self._phantomTransform_otherErr, self._phantomTransform_bestErr
        self._phantomTransform_bestIsFlipped = not self._phantomTransform_bestIsFlipped

        # The stored RMS of the flipped variant compares swapped endpoints (~rod length), not meaningful
        self._log("Flip applied (alternative phantom orientation) - check the rod segments")
        slicer.util.infoDisplay(
            "Alternative phantom orientation (flip) applied.\n"
            "Check 'Phantom_Segmentation'.\n"
            "Clicking again switches back.",
            windowTitle="Phantom Segmentator - Flip"
        )

    # ─── Remove voids ─────────────────────────────────────────────────────────

    def onButton_RemoveVoids(self):
        logging.info("[Phantom_Segmentator] Starting void removal.")

        volumeNode = self.currentVolumeNode
        if volumeNode is None:
            slicer.util.errorDisplay("No valid volume selected.")
            return

        segNode = slicer.mrmlScene.GetFirstNodeByName("Phantom_Segmentation")
        if segNode is None:
            slicer.util.errorDisplay(
                "Segmentation 'Phantom_Segmentation' not found.\n"
                "Please run 'Segment Phantom' first."
            )
            return

        try:
            from scipy.ndimage import binary_dilation
        except ImportError:
            slicer.util.errorDisplay("scipy not available - void removal not possible.")
            return

        HU_VOID_THR  = -250
        MARGIN_ITER  = 2
        struct       = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity

        huArray      = slicer.util.arrayFromVolume(volumeNode)
        voidMaskGlobal = huArray < HU_VOID_THR

        segmentation = segNode.GetSegmentation()
        n_segs       = segmentation.GetNumberOfSegments()
        total_removed, affected_segs = 0, 0

        for idx in range(n_segs):
            segId   = segmentation.GetNthSegmentID(idx)
            segName = segmentation.GetSegment(segId).GetName()

            segArray = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segId, volumeNode)
            if segArray is None:
                continue

            # Voxels within the segment below the HU threshold (= voids)
            bad_mask = (segArray > 0) & voidMaskGlobal
            if not bad_mask.any():
                continue

            # Grow the void region by 2 voxels (26-connectivity)
            bad_grown = binary_dilation(bad_mask, structure=struct, iterations=MARGIN_ITER)

            # Remove only voxels within the segment
            to_remove = bad_grown & (segArray > 0)
            n_bad  = int(bad_mask.sum())
            n_rem  = int(to_remove.sum())

            segArray[to_remove] = 0
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segArray, segNode, segId, volumeNode
            )

            total_removed += n_rem
            affected_segs += 1
            logging.info(
                f"[Phantom_Segmentator] Segment '{segName}': "
                f"{n_bad} void voxels, {n_rem} voxels removed (incl. {MARGIN_ITER}-voxel margin)."
            )

        self._updateButtonStates()

        if affected_segs == 0:
            slicer.util.infoDisplay(
                f"No voxels with HU < {HU_VOID_THR} found in the segments.",
                windowTitle="Phantom Segmentator - Voids"
            )
        else:
            slicer.util.infoDisplay(
                f"Void removal complete.\n"
                f"{affected_segs} segment(s) cleaned, {total_removed} voxels removed.",
                windowTitle="Phantom Segmentator - Voids"
            )
