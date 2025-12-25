# Task-Based EEG Dataset Search - Findings

**Date**: December 24, 2025 
**Purpose**: Identify appropriate task-based EEG datasets for EntPTC model testing

---

## Dataset 1: COG-BCI (Cognitive BCI Database)

**Source**: Nature Scientific Data (2023) 
**URL**: https://www.nature.com/articles/s41597-022-01898-y 
**DOI**: 10.1038/s41597-022-01898-y

### Key Details

**Participants**: 29 (11 Female, 18 Male), age 23.9 ± 3.2 years

**Sessions**: 3 sessions per participant, spaced 1 week apart

**Total Data**: Over 100 hours of open EEG data

**EEG Setup**:
- 64 active Ag-AgCl electrodes
- Standard 10-20 system
- Sampling rate: 500 Hz
- Impedances: < 25 kΩ
- Reference: Fpz

**Data Format**: BIDS format (.set and .fdt files)

### Tasks Included

1. **N-Back Task** ✅
 - Working memory task
 - Multiple difficulty levels
 - Elicits different mental workload states

2. **MATB-II (Multi-Attribute Task Battery)** ✅
 - Complex multitasking paradigm
 - Simulates operator workload
 - Multiple cognitive demands simultaneously

3. **PVT (Psychomotor Vigilance Task)** ✅
 - Sustained attention
 - Reaction time measurement
 - Vigilance monitoring

4. **Eriksen Flanker Task** ✅
 - Cognitive control
 - Attention and inhibition
 - Response conflict

5. **Resting State** ✅
 - Eyes open (EO)
 - Eyes closed (EC)
 - Beginning and end of session

### Why This Dataset Fits EntPTC Model

✅ **Working Memory (N-Back)**: Engages cognitive control and should drive regime transitions

✅ **Multitasking (MATB-II)**: Complex cognitive demands likely to elicit Regime II/III states

✅ **Sustained Attention (PVT)**: Tests temporal persistence and attractor stability

✅ **Cognitive Control (Flanker)**: Inhibition and conflict resolution should show distinct dynamics

✅ **Multiple Sessions**: Within-subject dynamics and learning effects

✅ **Time-Resolved**: Can track trajectory evolution during tasks

✅ **Large Sample**: 29 subjects × 3 sessions = 87 total sessions

✅ **Open Access**: Publicly available for download

### Access Information

**Repository**: OpenNeuro or direct from authors 
**Format**: BIDS-compliant 
**License**: Open access 
**Download**: Available through data repository links in paper

---

## Dataset Evaluation for EntPTC

### Requirements from Christopher's Instructions

| Requirement | COG-BCI Status |
|-------------|----------------|
| Task-based EEG | ✅ YES - 4 different tasks |
| Working memory | ✅ YES - N-Back task |
| Sustained attention | ✅ YES - PVT |
| Cognitive control | ✅ YES - Flanker task |
| State transitions | ✅ YES - Multiple task conditions |
| Time-resolved | ✅ YES - Continuous EEG during tasks |
| NOT passive resting state | ✅ YES - Active cognitive tasks |
| Exercises regime structure | ✅ LIKELY - Multiple cognitive demands |

### Expected Regime Engagement

**N-Back Task**:
- Low load (1-back): Likely Regime I (local stabilized)
- Medium load (2-back): Likely Regime II (transitional)
- High load (3-back): Potentially Regime III (global experience)

**MATB-II**:
- Complex multitasking should engage Regime II/III
- High cognitive demand and coordination

**PVT**:
- Sustained attention may show Regime I → II transitions
- Vigilance decrement over time

**Flanker**:
- Conflict trials may elicit brief Regime II states
- Cognitive control engagement

---

## Additional Datasets to Consider

### Dataset 2: PhysioNet Multimodal N-Back Dataset

**URL**: https://physionet.org/content/multimodal-nback-music/ 
**Focus**: Working memory with music modulation 
**Advantage**: Specifically designed for n-back research

### Dataset 3: OpenNeuro Task-Based Collections

**URL**: https://openneuro.org/ 
**Platform**: 1,565 public datasets 
**Search**: Can filter for EEG + cognitive tasks

### Dataset 4: Penn Memory Lab Data Portal

**URL**: https://memory.psych.upenn.edu/Data 
**Focus**: Cognitive electrophysiology 
**Tasks**: Memory, attention, cognitive control

---

## Recommendation

**PRIMARY DATASET**: COG-BCI Database

**Rationale**:
1. Comprehensive task battery (4 different tasks)
2. Specifically designed for mental state classification
3. Large sample size (29 subjects, 87 sessions)
4. High-quality EEG (64 channels, 500 Hz)
5. BIDS-compliant format (easy to process)
6. Open access and well-documented
7. Published in Nature Scientific Data (peer-reviewed)
8. Already used for BCI research (validated)

**Next Steps**:
1. Download COG-BCI dataset
2. Focus on N-Back and MATB-II tasks first
3. Implement toroidal dynamics analysis
4. Apply geometric falsifiability criteria
5. Compare across task conditions and difficulty levels

---

**Status**: Dataset identified and validated 
**Ready for**: Implementation and analysis
