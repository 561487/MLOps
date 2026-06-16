import { Drawer, Spin, } from 'antd';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getNodeInfoCommon, getNodeRelationCommon } from '../../api/commonPipeline';
import Loading from '../../components/Loading/Loading';
import { getParam } from '../../util';
import NodeDetail from './NodeDetail';
import RelationDiagram from './TreePlusDiagram';
import { ILayoutConfig, INodeDetailItem, INodeItem } from './TreePlusInterface';
import './TreePlus.less';
import { useTranslation } from 'react-i18next';

const DAG_REFRESH_INTERVAL = 3000;

interface IProps {
	id?: string | number | undefined | null;
	isCollapsed?: boolean;
}

export default function TreePlus(props: IProps) {
	const [loading, setLoading] = useState(false);
	const [visableDrawer, setVisableDrawer] = useState(false)
	const [isNoData, setIsNoData] = useState(false)
	const [nodeDetail, setNodeDetail] = useState<INodeDetailItem[]>([])
	const [loadingDetail, setLoadingDetail] = useState(false)
	const [layoutConfig, setLayoutConfig] = useState<ILayoutConfig>()
	const { t } = useTranslation();

	const treeDataRef = useRef<INodeItem[]>()

	const [relationDiagram, _setRelationDiagram] = useState<RelationDiagram>();
	const relationDiagramRef = useRef(relationDiagram);
	const setRelationDiagram = (data: RelationDiagram): void => {
		relationDiagramRef.current = data;
		_setRelationDiagram(data);
	};

	const handleClickNode = useCallback((node: any) => {
		// console.log(node)
		const currentNode = relationDiagramRef.current?.dataMap?.get(node.key)
		// console.log('currentNode', currentNode);

		setNodeDetail([])
		setLoadingDetail(true)
		setVisableDrawer(true)
		getNodeInfoCommon(currentNode?.detail_url || '').then(res => {
			// console.log(res.data.result.detail);
			const detail = res.data.result.detail
			setNodeDetail(detail)
			setLoadingDetail(false)
		}).catch(() => {
			setLoadingDetail(false)
		})
	}, [])

	const refreshDagAndLayout = useCallback((backurl = getParam('backurl') || '') => {
		if (backurl) {
			getNodeRelationCommon(backurl).then(res => {
				const dag = res.data.result.dag || []
				const layout = res.data.result.layout || {}
				const rd = relationDiagramRef.current
				setIsNoData(!dag.length)
				if (dag.length && rd) {
					treeDataRef.current = dag
					rd.updateNodes(dag)
				}
				setLayoutConfig(layout)
			})
		}
	}, [])

	const fetchBloodRelationData = useCallback((url: string) => {
		const rd = relationDiagramRef.current
		if (rd) {
			if (!!url) {
				setLoading(true);
				getNodeRelationCommon(url)
					.then((res) => {
						const dag = res.data.result.dag || []
						const layout = res.data.result.layout || {}
						treeDataRef.current = dag

						if (!dag.length) {
							setIsNoData(true)
						} else {
							setIsNoData(false)
						}
						rd.initData(dag);
						rd.handleNodeClick = handleClickNode;
						rd.loadingStart = () => {
							setLoading(true)
						}
						rd.loadingEnd = () => {
							setLoading(false)
						}
						setLayoutConfig(layout)
					})
					.catch((err) => {
						// console.log(err);
						setIsNoData(true)
					})
					.finally(() => {
						setLoading(false);
					});
			} else {
				rd.initData([])
				setIsNoData(true)
			}
		}
	}, [handleClickNode]);

	useEffect(() => {
		const target = new RelationDiagram({
			containerId: 'd3Container',
			mainViewId: 'd3MainView',
			margin: 16,
		});
		setRelationDiagram(target);

		const resizeIframe = document.getElementById('resizeIframe') as HTMLIFrameElement;
		if (resizeIframe.contentWindow) {
			resizeIframe.contentWindow.onresize = () => {
				setTimeout(() => {
					// console.log(relationDiagramRef.current);
					relationDiagramRef.current?.reSize();
				}, 1000);
			};
		}
	}, []);

	useEffect(() => {
		if (relationDiagram) {
			const backurl = getParam('backurl') || ''
			fetchBloodRelationData(backurl)
			if (backurl) {
				const timer = window.setInterval(() => {
					refreshDagAndLayout(backurl)
				}, DAG_REFRESH_INTERVAL)
				return () => {
					window.clearInterval(timer)
				}
			}
		}

	}, [fetchBloodRelationData, refreshDagAndLayout, relationDiagram]);

	return (
		<div className="p-r h100" id="fullContainer">
			<iframe
				id="resizeIframe"
				title="resize-listener"
				src=""
				frameBorder="0"
				className="p-a z-1"
				style={{ width: '100%', height: '100%' }}
			/>
			{
				loading ? <div className="p-a w100 h100 d-f ac jc mark z999 fadein">
					<Spin spinning={loading} indicator={<Loading />}>
						<div />
					</Spin>
				</div> : null
			}
			<Drawer
				title="节点详情"
				width={800}
				closable={false}
				onClose={() => { setVisableDrawer(false) }}
				visible={visableDrawer}
				className="nodedetail-wapper"
			>
				<Spin spinning={loadingDetail}>
					<NodeDetail data={nodeDetail} />
				</Spin>
			</Drawer>

			{
				isNoData ? <div className="p-a w100 h100 d-f ac jc ta-c z1">
					<div>
						<div><img className="w320" src={require('../../images/workData.png')} alt="" /></div>
						<div className="fs22">{t('暂无数据')}</div>
					</div>
				</div> : null
			}
			<div className="d-f fd-c h100 ov-h">
				<div className="tree-header">
					<div className="p16 d-f jc-b ac">
						<div className="d-f ac">
							<span className="icon-custom" dangerouslySetInnerHTML={{ __html: layoutConfig?.icon || '' }}></span>
							<span className="ml8 fs18">{layoutConfig?.title}</span>
						</div>
						<div>
							{
								(layoutConfig?.right_button || []).map(button => {
									return <div onClick={() => {
										window.open(button.url, "blank")
									}} className="c-text-w ml8 btn-ghost d-il">{button.label}</div>
								})
							}
						</div>
					</div>
					<div className="header-detail p16 d-f">
						{
							(layoutConfig?.detail || []).map(detail => {
								return <div className="flex1 mr48 header-detail-item">
									{
										detail.map(group => {
											return <div className="pb2">{group.label}：{group.value}</div>
										})
									}
								</div>
							})
						}
					</div>
				</div>
				<div id="d3Container" className="flex1">
					<svg id="d3MainView" />
				</div>
			</div>
		</div>
	);
}
