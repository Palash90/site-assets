<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <html>
            <body>
                <h2>Game List</h2>
                <table border="1">
                    <tr bgcolor="#9acd32">
                        <th>Link</th>
                    </tr>
                    <xsl:for-each select="urlset/url">
                        <tr>
                            <td><xsl:value-of select="loc"/></td>
                        </tr>
                    </xsl:for-each>
                </table>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>
