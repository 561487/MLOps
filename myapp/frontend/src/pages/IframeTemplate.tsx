import React, { useState } from 'react'
import { IAppMenuItem } from '../api/interface/kubeflowInterface'
import { getParam } from '../util'
import { THEME_STORAGE_KEY, TThemeType, getInitialTheme, isThemeType } from '../theme'
import globalConfig from '../global.config'


function getCurrentTheme(): TThemeType {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
    return isThemeType(storedTheme) ? storedTheme : getInitialTheme(globalConfig.theme)
}

function withThemeParam(url?: string) {
    if (!url) return url

    try {
        const targetUrl = new URL(url, window.location.origin)
        targetUrl.searchParams.set('theme', getCurrentTheme())
        return targetUrl.toString()
    } catch (error) {
        return url
    }
}

export default function IframeTemplate(props?: IAppMenuItem) {
    const [url, setUrl] = useState(withThemeParam(getParam('url') || props?.url))
    return (
        <>
            <iframe id="_frontendAppCustomFrame_"
                src={url}
                allowFullScreen
                allow="microphone;camera;midi;encrypted-media;"
                className="w100 h100 fade-in"
                style={{ border: 0, display: 'block'}}>
            </iframe>
        </>
    )
}
