import { SettingOutlined, UserOutlined } from '@ant-design/icons';
import React from 'react';
import { IRouterConfigPlusItem } from './api/interface/baseInterface';
import { IAppMenuItem } from './api/interface/kubeflowInterface';
import Page404 from './pages/Page404';
import Cookies from 'js-cookie'
import LoadingStar from './components/LoadingStar/LoadingStar';
const userName = Cookies.get('myapp_username')
const isAdmin = userName === 'admin'

const LoadingComponent = () => {
    return <div className="d-f ac jc w100 h100">
        <LoadingStar />
    </div>
}

const lazy2Compont = (factory: () => Promise<{
    default: () => JSX.Element;
}>, props?: any) => {
    const Component = React.lazy(() => factory())
    const id = Math.random().toString(36).substring(2)
    return <React.Suspense key={id} fallback={<LoadingComponent />}>
        <Component {...props} />
    </React.Suspense>
}

export const securitySettingConfig: IRouterConfigPlusItem[] = [
    {
        path: '/security',
        name: 'security',
        title: '安全设置',
        isLocalMenu: true,
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        children: [
            {
                path: '/security/userList',
                title: '用户列表',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/users/list/?_flt_2_username=' })
            },
            {
                path: '/security/roleList',
                title: '角色列表',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/roles/list/?_flt_2_name=' })
            },
            {
                path: '/security/statistics',
                title: '用户统计',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/userstatschartview/chart/' })
            },
            {
                path: '/security/permissions',
                title: '权限列表',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: 'permissions/list/' })
            },
            {
                path: '/security/view',
                title: '视图列表',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/viewmenus/list/' })
            },
            {
                path: '/security/permissionsOnView',
                title: '权限视图关系',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/permissionviews/list/' })
            },
            {
                path: '/security/log',
                title: '日志列表',
                menu_type: 'iframe',
                icon: <SettingOutlined style={{ marginRight: 8 }} />,
                element: lazy2Compont(() => import("./pages/IframeTemplate"), { url: '/log_modelview/api/' })
            },
        ]
    },
]

export const innerDynamicRouterConfig: IRouterConfigPlusItem[] = [
    {
        path: '/dataSearch',
        title: '数据查询',
        key: 'data_search',
        icon: '',
        menu_type: 'innerRouter',
        isCollapsed: true,
        element: lazy2Compont(() => import("./pages/DataSearch/DataSearch"))
    },
    {
        path: '/commonRelation',
        title: '通用关系图',
        element: lazy2Compont(() => import("./pages/CommonPipeline/DWStandard") as any)
    },    // 模型市场相关路由 — 路径对应菜单 hierarchy: service > model_market_group
    {
        path: '/service/model_market_group/model_market_vision',
        title: '视觉模型',
        key: 'model_market_vision',
        menu_type: 'innerRoute',
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelMarketCategory") as any)
    },
    {
        path: '/service/model_market_group/model_market_audio',
        title: '语音模型',
        key: 'model_market_audio',
        menu_type: 'innerRoute',
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelMarketCategory") as any)
    },
    {
        path: '/service/model_market_group/model_market_nlp',
        title: '自然语言模型',
        key: 'model_market_nlp',
        menu_type: 'innerRoute',
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelMarketCategory") as any)
    },
    {
        path: '/service/model_market_group/model_market_multimodal',
        title: '多模态模型',
        key: 'model_market_multimodal',
        menu_type: 'innerRoute',
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelMarketCategory") as any)
    },
    {
        path: '/service/model_market_group/model_market_llm',
        title: '大模型',
        key: 'model_market_llm',
        menu_type: 'innerRoute',
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelMarketCategory") as any)
    },
    {
        path: '/service/model_market_group/detail/:modelId',
        title: '模型详情',
        key: 'model_market_detail',
        menu_type: 'innerRoute',
        hidden: true,
        element: lazy2Compont(() => import("./pages/ModelMarket/ModelDetail") as any)
    },
    {
        path: '/service/model_market_group/service/:marketServiceId',
        title: '推理服务',
        key: 'model_market_service',
        menu_type: 'innerRoute',
        hidden: true,
        element: lazy2Compont(() => import("./pages/ModelMarket/ServiceDetail") as any)
    },
]

const innerDynamicRouterConfigMap = innerDynamicRouterConfig.reduce((pre, next) => ({
    ...pre,
    [next.key || '']: next
}), {}) as Record<string, IRouterConfigPlusItem>;

export const routerConfigPlus: IRouterConfigPlusItem[] = [
    {
        path: '/',
        index: true,
        element: lazy2Compont(() => import("./pages/Home/Home") as any)
    },    // 重定向：点击模型市场父级菜单时默认跳转到视觉
    {
        path: '/service/model_market_group',
        element: lazy2Compont(() => import("./pages/ModelMarket/RedirectToHome") as any)
    },
    {
        path: '/showData',
        title: '数据展示',
        element: lazy2Compont(() => import("./pages/ShowData"))
    },
    {
        path: '/showOutLink',
        title: '外链',
        element: lazy2Compont(() => import("./pages/IframeTemplate"))
    },
    ...innerDynamicRouterConfig,
    {
        path: '/user',
        icon: <UserOutlined style={{ fontSize: 18 }} />,
        element: lazy2Compont(() => import("./pages/UserCenter") as any)
    },
    { path: '*', element: <Page404 /> },
]

// if (isAdmin) {
//     routerConfigPlus.push(...securitySettingConfig)
// }

export const formatRoute = (data: IAppMenuItem[]): IRouterConfigPlusItem[] => {
    // console.log(data)
    const dfs = (data: IAppMenuItem[], chain?: IAppMenuItem[]): IRouterConfigPlusItem[] => {
        const res: IRouterConfigPlusItem[] = []
        for (let i = 0; i < data.length; i++) {
            const item = data[i];
            const currentChain = chain ? [...chain, item] : [item]
            const currentPath = currentChain.map(item => item.name)
            const routeProps = {
                isCollapsed: true,
                ...item,
                breadcrumbs: currentChain.map(item => item.title)
            }
            if (item.children?.length > 0) {
                res.push({
                    ...item,
                    path: `/${currentPath.join('/')}`,
                    children: dfs(item.children, currentChain),
                    // element: lazy2Compont(() => import("./pages/Home"), { ...routeProps }),
                })
            } else {
                const ADUGTemplateComponent = lazy2Compont(() => import("./pages/ADUGTemplate"), { ...routeProps, related: item.related })
                const OutLinkComponent = lazy2Compont(() => import("./pages/OutLinkTemplate"), { ...routeProps, related: item.related })
                const IframeComponent = lazy2Compont(() => import("./pages/IframeTemplate"), { ...routeProps, related: item.related })

                if (item.menu_type === 'out_link') {
                    res.push({
                        ...item,
                        path: `/${currentPath.join('/')}`,
                        element: OutLinkComponent
                    })
                } else if (item.menu_type === 'innerRoute') {
                    const page = innerDynamicRouterConfigMap[item.name]
                    res.push({
                        ...item,
                        path: `/${currentPath.join('/')}`,
                        element: page ? innerDynamicRouterConfigMap[item.name].element : ADUGTemplateComponent
                    })
                } else if (item.menu_type === 'iframe') {
                    res.push({
                        ...item,
                        path: `/${currentPath.join('/')}`,
                        element: IframeComponent
                    })
                } else {
                    res.push({
                        ...item,
                        path: `/${currentPath.join('/')}`,
                        element: ADUGTemplateComponent
                    })
                }

                if (item.related && item.related.length) {
                    for (let i = 0; i < item.related.length; i++) {
                        const relatedItem = item.related[i];
                        res.push({
                            ...relatedItem,
                            path: `/${currentPath.join('/')}/${relatedItem.name}`,
                            element: lazy2Compont(() => import("./pages/ADUGTemplate"), {
                                isCollapsed: true,
                                ...relatedItem,
                                isSubRoute: true,
                                model_name: item.model_name,
                                breadcrumbs: [...currentChain.map(item => item.title), relatedItem.title]
                            }),
                        })
                    }
                }
            }
        }
        return res
    }
    const resData = dfs(data)
    return resData
}

export const getDefaultOpenKeys = (data: IRouterConfigPlusItem[]) => {
    const openKeys: string[] = []
    const quene = [...data]
    while (quene.length) {
        const item = quene.shift()
        if (item?.isExpand && item.path) {
            openKeys.push(item.path)
        }
        if (item?.children && item.children[0]) {
            quene.push(...item.children)
        }
    }
    return openKeys
}
