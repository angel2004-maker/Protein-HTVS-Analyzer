import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import AlignIO
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, PDBIO
import py3Dmol
import tempfile
import io
import re

# ============ 页面配置 ============
st.set_page_config(
    page_title='功能蛋白智能分析系统',
    layout='wide',
    menu_items={'About': '功能蛋白智能分析系统 - 面向蛋白药物设计的智能分析平台'}
)

st.title('功能蛋白智能分析系统')
st.caption('面向功能蛋白药物设计与开发的智能分析平台')

# ============ 氨基酸分类 ============
HYDROPHOBIC = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
HYDROPHILIC = ['S', 'T', 'N', 'Q', 'Y', 'C']
POSITIVE = ['K', 'R', 'H']
NEGATIVE = ['D', 'E']

# Kyte-Doolittle 疏水性标度
KD_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}


def classify_amino_acids(sequence):
    """统计氨基酸分类"""
    seq = sequence.upper()
    counts = {
        '疏水氨基酸': sum(seq.count(aa) for aa in HYDROPHOBIC),
        '亲水氨基酸': sum(seq.count(aa) for aa in HYDROPHILIC),
        '带正电氨基酸': sum(seq.count(aa) for aa in POSITIVE),
        '带负电氨基酸': sum(seq.count(aa) for aa in NEGATIVE)
    }
    return counts


def calculate_kd_plot(sequence, window=9):
    """计算 Kyte-Doolittle 滑动窗口疏水性"""
    seq = sequence.upper()
    scores = []
    for i in range(len(seq) - window + 1):
        window_seq = seq[i:i + window]
        score = sum(KD_HYDROPATHY.get(aa, 0) for aa in window_seq) / window
        scores.append(score)
    return scores


def evaluate_protein_properties(sequence):
    """评估蛋白质理化性质"""
    seq = sequence.upper()
    counts = classify_amino_acids(seq)
    total = len(seq)

    hydrophobic_ratio = counts['疏水氨基酸'] / total
    kd_scores = calculate_kd_plot(seq, window=9)
    max_kd = max(kd_scores) if kd_scores else 0

    results = {
        'risk_level': 'low',
        '稳定性': '良好',
        '溶解性': '良好',
        '聚沉风险': '低',
        '建议': []
    }

    if hydrophobic_ratio > 0.45 or max_kd > 2.0:
        results['risk_level'] = 'high'
        results['稳定性'] = '高风险'
        results['溶解性'] = '极低'
        results['聚沉风险'] = '极高'
        results['建议'].append('该候选序列整体疏水性超标，极易在表达过程中形成包涵体沉淀，提纯难度极大。作为工业候选药物的风险不可控，强烈建议直接淘汰，或退回 ProteinMPNN 重新设计。')
    elif hydrophobic_ratio > 0.38 or max_kd > 1.6:
        results['risk_level'] = 'medium'
        results['稳定性'] = '一般'
        results['溶解性'] = '较差'
        results['聚沉风险'] = '中'
        results['建议'].append('疏水氨基酸比例偏高，存在一定聚沉风险。建议进行优化或增加分子伴侣共表达。')
    else:
        results['建议'].append('该序列理化性质良好，表面亲水性合格，适合作为候选蛋白药物进入大肠杆菌表达等下游湿实验验证。')

    return results


# ============ 结构比对工具函数 ============
def parse_structure_from_text(text, name='structure'):
    """从 PDB/CIF 文本内容解析结构对象"""
    text_content = text if isinstance(text, str) else text.decode('utf-8')
    is_cif = text_content.strip().startswith('data_') or 'loop_' in text_content[:100] or '_entry.' in text_content[:200]

    try:
        if is_cif:
            parser = MMCIFParser()
        else:
            parser = PDBParser(PERMISSIVE=True)
        structure = parser.get_structure(name, io.StringIO(text_content))
        return structure, None
    except Exception as e:
        return None, f'结构解析失败: {e}'


def get_ca_atoms(structure, chain_id=None):
    """提取结构中的 Cα 原子"""
    atoms = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] == ' ' and residue.resname not in ['MSE', 'MET']:
                    if 'CA' in residue:
                        atoms.append(residue['CA'])
    return atoms


def extract_pdb_block(structure, chain_id=None):
    """将结构对象转回 PDB 文本块"""
    io_obj = io.StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(io_obj)
    return io_obj.getvalue()


def align_structures(mobile_struct, reference_struct, chain_id=None):
    """
    使用 Bio.PDB Superimposer 将 mobile 对齐到 reference
    返回: (rmsd, mobile_aligned_pdb_text, ref_pdb_text, info)
    """
    ref_atoms = get_ca_atoms(reference_struct, chain_id)
    mob_atoms = get_ca_atoms(mobile_struct, chain_id)

    if len(ref_atoms) < 3 or len(mob_atoms) < 3:
        return None, None, None, 'Cα 原子数量不足，无法比对'

    min_len = min(len(ref_atoms), len(mob_atoms))
    ref_ca = ref_atoms[:min_len]
    mob_ca = mob_atoms[:min_len]

    superimposer = Superimposer()
    superimposer.set_atoms(ref_ca, mob_ca)
    rmsd = superimposer.rms

    # 对 mobile 的完整拷贝应用变换（所有原子一起转，避免骨架断裂）
    mobile_copy = mobile_struct.copy()
    superimposer.apply(list(mobile_copy.get_atoms()))
    mob_pdb = extract_pdb_block(mobile_copy)

    ref_pdb = extract_pdb_block(reference_struct)

    info = f'参考结构 {len(ref_atoms)} 个 Cα，移动结构 {len(mob_atoms)} 个 Cα，比对用了 {min_len} 个 Cα'
    return rmsd, mob_pdb, ref_pdb, info


# ============ 创建标签页 ============
tab1, tab2, tab3, tab4 = st.tabs(['理化性质扫描', '保守性分析', '3D结构可视化', '结构比对'])

# ============================================================
# 结构比对看板
# ============================================================
with tab4:
    st.subheader('AI 设计结构 vs 野生型晶体结构 比对')
    st.caption('基于 Cα 原子进行 Needleman-Wunsch 序列对齐后三维空间叠加')

    col_ref, col_mob = st.columns(2)

    with col_ref:
        ref_file = st.file_uploader(
            '上传原始结构（野生型晶体结构）',
            type=['pdb', 'cif'],
            key='ref_struct'
        )

    with col_mob:
        mob_file = st.file_uploader(
            '上传待比对结构（AlphaFold3 预测设计结构）',
            type=['pdb', 'cif'],
            key='mob_struct'
        )

    if ref_file and mob_file:
        ref_text = ref_file.read().decode('utf-8')
        mob_text = mob_file.read().decode('utf-8')

        ref_struct, ref_err = parse_structure_from_text(ref_text, 'reference')
        mob_struct, mob_err = parse_structure_from_text(mob_text, 'mobile')

        if ref_err:
            st.error(ref_err)
        if mob_err:
            st.error(mob_err)

        if ref_struct and mob_struct:
            with st.spinner('正在进行三维空间对齐...'):
                rmsd, mob_aligned_pdb, ref_pdb, info = align_structures(mob_struct, ref_struct)

            if rmsd is not None:
                col_metric = st.columns([1, 2, 1])
                with col_metric[1]:
                    st.metric(
                        label='Cα RMSD（Å）',
                        value=f'{rmsd:.3f}',
                        help='均方根误差，越小表示结构越接近'
                    )
                    st.caption(f'ℹ️ {info}')

                st.divider()
                st.subheader('三维结构重叠可视化')

                # 控件定义在视图渲染之前，依赖 Streamlit 值变自动 rerun
                ref_color = st.selectbox('原始结构颜色', ['灰色', '白色', '绿色', '蓝色'], key='ref_color')
                mob_color = st.selectbox('AF3预测结构颜色', ['海洋蓝', '红色', '橙色', '紫色'], key='mob_color')
                ref_style = st.radio('原始结构样式', ['卡通图', '球棍模型', '表面图'], key='ref_style')
                mob_style = st.radio('AF3预测结构样式', ['卡通图', '球棍模型', '表面图'], key='mob_style')

                ref_color_map = {'灰色': 'gray', '白色': 'white', '绿色': 'green', '蓝色': 'blue'}
                mob_color_map = {'海洋蓝': 'spectrum', '红色': 'red', '橙色': 'orange', '紫色': 'purple'}
                ref_style_map = {'卡通图': 'cartoon', '球棍模型': 'stick', '表面图': 'surface'}
                mob_style_map = {'卡通图': 'cartoon', '球棍模型': 'stick', '表面图': 'surface'}

                ref_style_val = ref_style_map[ref_style]
                mob_style_val = mob_style_map[mob_style]
                ref_opacity = 0.7 if ref_style == '卡通图' else 1.0

                view = py3Dmol.view(width=700, height=500)
                view.addModel(ref_pdb, 'pdb')
                view.addModel(mob_aligned_pdb, 'pdb')
                view.setStyle({'model': 0}, {ref_style_val: {'color': ref_color_map[ref_color], 'opacity': ref_opacity}})
                view.setStyle({'model': 1}, {mob_style_val: {'color': mob_color_map[mob_color]}})
                view.zoomTo()
                st.components.v1.html(view._make_html(), height=520, scrolling=False)
            else:
                st.error(info)
    else:
        st.info('请同时上传原始结构和 AF3 预测结构开始比对分析')

# ============================================================
# 基础看板
# ============================================================
with tab1:
    col_file, col_text = st.columns(2)

    with col_file:
        st.markdown('**方式一：上传 FASTA 文件**')
        fasta_file = st.file_uploader('选择 FASTA 文件', type=['fasta', 'fa'], key='fasta_basic')

    with col_text:
        st.markdown('**方式二：直接输入序列**')
        manual_seq = st.text_area(
            '粘贴蛋白质序列（单行）',
            placeholder='例如：DVQLVESGGGSVQAGGSLRLSCAASGYIASINYLGWFRQAPGKEREGVAAVSPAGGTPYYADSVKGRFTVSLDNAENTVYLQMNSLKPEDTALYYCAAARQGWYIPLNSYGYNYWGQGTQVTVS',
            height=120,
            key='manual_seq'
        )

    sequence = None

    if fasta_file:
        fasta_text = fasta_file.read().decode('utf-8')
        records = list(SeqIO.parse(io.StringIO(fasta_text), 'fasta'))
        if records:
            record = records[0]
            sequence = str(record.seq)
            st.success(f'已加载序列：{record.id}（长度：{len(sequence)} 个氨基酸）')

    elif manual_seq:
        seq_clean = re.sub(r'[^A-Za-z]', '', manual_seq.strip())
        if len(seq_clean) >= 10:
            sequence = seq_clean
            st.success(f'已加载手动输入序列（长度：{len(sequence)} 个氨基酸）')
        elif seq_clean:
            st.warning('序列过短，请输入至少 10 个氨基酸')

    if sequence:
        counts = classify_amino_acids(sequence)
        total = sum(counts.values())
        hydrophobic_ratio = counts['疏水氨基酸'] / total
        kd_scores = calculate_kd_plot(sequence, 9)
        max_kd = max(kd_scores) if kd_scores else 0
        properties = evaluate_protein_properties(sequence)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader('氨基酸分类统计')

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    text=[f'{v} ({v/total*100:.1f}%)' for v in counts.values()],
                    textposition='auto',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )
            ])
            fig_bar.update_layout(
                yaxis_title='氨基酸数量',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            if hydrophobic_ratio > 0.45 or max_kd > 2.0:
                st.error(f'⚠️ 疏水氨基酸占比 {hydrophobic_ratio*100:.1f}%，滑动窗口最高 {max_kd:.2f}，远超安全阈值，强烈建议淘汰！')
            elif hydrophobic_ratio > 0.38 or max_kd > 1.6:
                st.warning(f'⚠️ 疏水氨基酸占比 {hydrophobic_ratio*100:.1f}%，滑动窗口最高 {max_kd:.2f}，有一定聚沉风险')

        with col2:
            st.subheader('亲疏水性滑动窗口分析')
            st.caption('Kyte-Doolittle 疏水性图谱（正值=疏水，负值=亲水）')

            window_size = st.slider('滑动窗口大小', 5, 21, 9, key='window_slider')
            kd_scores = calculate_kd_plot(sequence, window_size)
            positions = list(range(1, len(kd_scores) + 1))

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=positions,
                y=kd_scores,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#2E86AB'),
                name='疏水性得分'
            ))
            fig_line.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
            fig_line.add_hrect(y0=1.6, y1=max(kd_scores) + 0.5,
                               fillcolor='red', opacity=0.1, annotation_text='潜在跨膜区')
            fig_line.update_layout(
                xaxis_title='氨基酸位置',
                yaxis_title='疏水性得分',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        st.subheader('理化性质评估报告')

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric('稳定性', properties['稳定性'])
        with col_b:
            st.metric('溶解性', properties['溶解性'])
        with col_c:
            st.metric('聚沉风险', properties['聚沉风险'])

        if properties['risk_level'] == 'high':
            st.markdown('<div style="background-color:#ffcccc;padding:12px;border-radius:5px;color:#cc0000;font-weight:bold">⚠️ 该候选序列整体疏水性超标（>45%），极易在表达过程中形成包涵体沉淀，提纯难度极大。作为工业候选药物的风险不可控，强烈建议直接淘汰，或退回 ProteinMPNN 重新设计。</div>', unsafe_allow_html=True)
        elif properties['risk_level'] == 'medium':
            st.markdown('<div style="background-color:#fff3cd;padding:12px;border-radius:5px;color:#856404;font-weight:bold">⚠️ 疏水氨基酸比例偏高（>38%），存在一定聚沉风险。建议进行优化或增加分子伴侣共表达。</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color:#d4edda;padding:12px;border-radius:5px;color:#155724;font-weight:bold">✅ 该序列理化性质良好，表面亲水性合格，适合作为候选蛋白药物进入大肠杆菌表达等下游湿实验验证。</div>', unsafe_allow_html=True)
    else:
        st.info('请上传 FASTA 文件或直接输入序列开始分析')

    # ============================================================
    # 高通量批量筛查（与单序列分析并列，独立运行）
    # ============================================================
    st.divider()
    st.subheader('高通量理化性质批量筛查')
    st.caption('前置漏斗：批量评估多条序列，筛选出值得提交 AlphaFold3 进行结构预测的候选序列')

    batch_fasta = st.file_uploader('上传包含多条序列的 .fasta 文件（用于批量筛查）', type=['fasta', 'fa'], key='batch_fasta')

    if batch_fasta:
        batch_text = batch_fasta.read().decode('utf-8')
        batch_records = list(SeqIO.parse(io.StringIO(batch_text), 'fasta'))

        if len(batch_records) < 2:
            st.warning('批量筛查至少需要 2 条序列')
        else:
            st.info(f'正在分析 {len(batch_records)} 条序列，请稍候...')

            batch_results = []
            for record in batch_records:
                seq = str(record.seq).upper()
                seq_id = record.id if record.id else f'seq_{len(batch_results)+1}'
                counts = classify_amino_acids(seq)
                total = sum(counts.values()) if counts else len(seq)
                hyd_ratios = counts['疏水氨基酸'] / total if total > 0 else 0

                gravys = calculate_kd_plot(seq, window=len(seq))
                gravys = [s for s in gravys if s != 0]
                gravy = sum(gravys) / len(gravys) if gravys else 0

                status = '❌ 聚沉淘汰' if hyd_ratios > 0.45 else '✅ 理化合格'
                batch_results.append({
                    '序列ID': seq_id[:40],
                    '长度': len(seq),
                    '疏水占比(%)': round(hyd_ratios * 100, 2),
                    'GRAVY得分': round(gravy, 4),
                    '风控判定': status
                })

            df = pd.DataFrame(batch_results)

            st.markdown('**序列排行榜**')

            html_table = df.to_html(index=False, escape=False, classes='styled-table')
            styled_html = f'<style>.styled-table{{border-collapse:collapse;width:100%;font-size:14px}}.styled-table th,.styled-table td{{padding:8px;border-bottom:1px solid #ddd;text-align:center}}.styled-table tr:nth-child(even){{background:#f9f9f9}}</style>' + html_table.replace(
                '<td>❌ 聚沉淘汰</td>', '<td style="background-color:#ffcccc;color:#cc0000;font-weight:bold">❌ 聚沉淘汰</td>'
            ).replace(
                '<td>✅ 理化合格</td>', '<td style="background-color:#d4edda;color:#155724;font-weight:bold">✅ 理化合格</td>'
            )
            st.markdown(styled_html, unsafe_allow_html=True)

            pass_indices = [i for i, row in enumerate(batch_results) if '合格' in row['风控判定']]
            pass_records = [batch_records[i] for i in pass_indices]

            col_pass, col_dl = st.columns([1, 1])
            with col_pass:
                st.metric('理化合格候选数', len(pass_records), help='疏水占比 ≤45% 的序列')
            with col_dl:
                st.metric('待淘汰序列数', len(batch_records) - len(pass_records))

            if pass_records:
                pass_fasta = '\n'.join([f'>{r.id}\n{str(r.seq)}' for r in pass_records])
                st.download_button(
                    label='⬇️ 下载理化合格候选池 (.fasta)',
                    data=pass_fasta,
                    file_name='pass_candidates.fasta',
                    mime='text/plain',
                    key='dl_pass_fasta'
                )

# ============================================================
# 中级看板
# ============================================================
with tab2:
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.markdown('**方式一：上传比对文件**')
        aln_file = st.file_uploader('上传 .praln 或 .aln 比对文件', type=['praln', 'aln'], key='aln_file')

    with col_input2:
        st.markdown('**方式二：上传多条 FASTA 序列**')
        multi_fasta = st.file_uploader('上传包含多条序列的 FASTA 文件', type=['fasta', 'fa'], key='multi_fasta')

    # 处理比对文件
    sequences = None

    if aln_file:
        try:
            aln_text = aln_file.read().decode('utf-8')
            # 尝试解析为 clustal 格式
            alignment = AlignIO.read(io.StringIO(aln_text), 'clustal')
            sequences = [str(record.seq) for record in alignment]
            st.success(f'已加载比对结果：{len(sequences)} 条序列，长度 {alignment.get_alignment_length()}')
        except:
            st.error('无法解析比对文件，请确保格式正确')

    elif multi_fasta:
        fasta_text = multi_fasta.read().decode('utf-8')
        records = list(SeqIO.parse(io.StringIO(fasta_text), 'fasta'))
        if len(records) >= 2:
            sequences = [str(record.seq) for record in records]
            st.success(f'已加载 {len(sequences)} 条序列')
        else:
            st.warning('需要至少 2 条序列才能进行保守性分析')

    if sequences:
        # 检查序列长度是否一致
        seq_lengths = [len(seq) for seq in sequences]
        min_len = min(seq_lengths)
        max_len = max(seq_lengths)

        if max_len != min_len:
            st.warning(f'⚠️ 检测到序列长度不一致（{min_len} ~ {max_len}），将以最短序列（{min_len}）为基准计算保守性')
            seq_len = min_len
        else:
            seq_len = len(sequences[0])

        # 计算保守性
        conservation = []

        for i in range(seq_len):
            column = [seq[i] for seq in sequences]
            # 过滤 gap 和无效字符
            valid_column = [aa for aa in column if aa not in ['-', '.', ' ']]
            if not valid_column:
                cons_score = 0
            else:
                most_common = max(set(valid_column), key=valid_column.count)
                cons_score = valid_column.count(most_common) / len(valid_column)
            conservation.append(cons_score)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('保守性热力图')
            st.caption('颜色越深 = 保守度越高 = 功能关键位点')

            # 创建热力图
            fig_heat = go.Figure(data=go.Heatmap(
                z=[conservation],
                x=list(range(1, seq_len + 1)),
                colorscale=[
                    [0, '#FFFFFF'],
                    [0.33, '#FFE4B5'],
                    [0.66, '#FF8C00'],
                    [1, '#8B0000']
                ],
                colorbar=dict(title='保守度'),
                showscale=True
            ))
            fig_heat.update_layout(
                xaxis_title='氨基酸位置',
                height=200,
                yaxis=dict(showticklabels=False),
                margin=dict(l=0, r=0, t=10, b=30)
            )
            st.plotly_chart(fig_heat)

            # 解读
            high_cons = [i for i, c in enumerate(conservation) if c > 0.8]
            low_cons = [i for i, c in enumerate(conservation) if c < 0.4]

            st.markdown('**保守性解读：**')
            if high_cons:
                st.warning(f'🔴 高度保守位点 {len(high_cons)} 个（红色区域）：这些是功能关键位点，**改造时务必避开**')
            if low_cons:
                st.success(f'🟢 低保守位点 {len(low_cons)} 个（浅色区域）：这些是 AI 设计的**广阔天地**')

        with col2:
            st.subheader('保守性分布曲线')

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(1, seq_len + 1)),
                y=conservation,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#8B0000')
            ))
            fig_line.add_hline(y=0.8, line_dash='dash', line_color='red',
                              annotation_text='高保守阈值')
            fig_line.add_hline(y=0.4, line_dash='dash', line_color='green',
                              annotation_text='低保守阈值')
            fig_line.update_layout(
                xaxis_title='氨基酸位置',
                yaxis_title='保守度',
                height=400
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info('请上传比对文件或多条 FASTA 序列开始保守性分析')

# ============================================================
# 高级看板
# ============================================================
with tab3:
    pdb_file = st.file_uploader('上传 PDB 文件', type=['pdb'], key='pdb_file')

    if pdb_file:
        pdb_content = pdb_file.read().decode('utf-8')

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader('3D 结构视图')
            st.caption('鼠标拖拽旋转，滚轮缩放')

            # 创建 py3Dmol 视图
            view = py3Dmol.view(width=700, height=500)
            view.addModel(pdb_content, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.zoomTo()

            st.components.v1.html(view._make_html(), height=520, scrolling=False)

        with col2:
            st.subheader('显示选项')

            style_option = st.radio(
                '显示模式',
                ['卡通图', '球棍模型', '表面图', '带状图'],
                key='style_radio'
            )

            show_pocket = st.checkbox('标注潜在口袋位点', key='pocket_check')

            st.divider()
            st.markdown('**结构信息**')

            # 简单统计
            atom_count = pdb_content.count('\nATOM')
            st.metric('原子数量', atom_count)

            st.markdown('**口袋分析提示**')
            st.caption('口袋是药物分子结合的关键位置。通常位于：')
            st.caption('• 蛋白表面凹陷处')
            st.caption('• 两个结构域之间')
            st.caption('• 柔性环区附近')
    else:
        st.info('请上传 PDB 格式的蛋白质结构文件开始 3D 可视化')
        st.caption('可从 RCSB PDB 数据库 (rcsb.org) 下载 PDB 文件')
