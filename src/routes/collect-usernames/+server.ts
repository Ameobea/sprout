import { error, text, type RequestHandler } from '@sveltejs/kit';
import { JSDOM } from 'jsdom';

import { ADMIN_API_TOKEN } from '../../conf';
import { addUsernames } from '../add-usernames/addUsernames';

export const POST: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  }

  if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }

  try {
    const html = await fetch('https://myanimelist.net/users.php', {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
      },
    }).then((res) => res.text());
    const dom = new JSDOM(html);
    const usernames = [
      ...dom.window.document.querySelector('#content > table > tbody > tr > td:nth-child(1) > table > tbody').children,
    ]
      .flatMap((tr) => [...tr.children])
      .map((td) => td.children[0].children[0].innerHTML);
    if (!Array.isArray(usernames) || usernames.some((username) => typeof username !== 'string')) {
      console.error('Failed to parse usernames from the page.  HTML: ', html, usernames);
      error(500, 'Failed to parse usernames from the page');
    }

    const resText = await addUsernames(usernames);
    return text(resText);
  } catch (err) {
    console.error('Error fetching + processing MAL users page: ', err);
    error(500, 'Error fetching + processing MAL users page');
  }
};
